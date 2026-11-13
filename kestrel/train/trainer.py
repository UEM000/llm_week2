import json
import os
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import torch
from tqdm.auto import tqdm

from kestrel.config import Config
from kestrel.data.loader import make_train_loader
from kestrel.model.model import load_checkpoint, save_checkpoint
from kestrel.optim.build import build_optimizer
from kestrel.optim.schedule import wsd_lr
from kestrel.seed import seed_everything
from kestrel.train.evaluate import eval_loss
from kestrel.train.metrics import checkpoint_meta

LOG_EVERY = 10
FINETUNE_DENSE_LOG_STEPS = 50  # первые шаги дообучения логируем на каждом шаге
SPEED_WARMUP_STEPS = 5         # замер скорости: шаги 6-15 (после 5 разогревочных)
SPEED_LAST_STEP = 15


def final_path(ckpt_path) -> Path:
    """Файл с итоговыми весами в формате чекпоинта {"model", "meta"}: <ckpt>.final.pt рядом с чекпоинтом."""
    return Path(ckpt_path).with_suffix(".final.pt")


def is_done(ckpt_path) -> bool:
    return final_path(ckpt_path).exists()


def read_log(log_path) -> list[dict]:
    if log_path is None or not Path(log_path).exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_log(log_path, rows: list[dict]) -> None:
    with open(log_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _sync(dev) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()


def train(
    model,
    kind: Literal["adamw", "muon"],
    stage: Literal["pretrain", "finetune"],
    cfg: Config,
    data_dir,
    ckpt_path,
    init_from: str | None = None,
    log_path: str | None = None,
    num_workers: int = 2,
) -> dict:
    """Обучает модель одну стадию.

    Возвращает {'steps', 'train_loss', 'val_loss', 'step_sec', 'data_wait_sec', 'final_path', 'log_path'}.

    - pretrain: данные pretrain_*.bin, YaRN scale = cfg.pretrain.yarn_scale;
      finetune: данные long_*.bin, model.set_yarn_scale(cfg.finetune.yarn_scale), состояние оптимизатора всегда новое.
    - init_from: чекпоинт для старта (init.pt, ckpt_A.pt, ckpt_M.pt: {"model", "meta"}); игнорируется при resume.
    - Батч = micro_batch_size * grad_accum; градиент накапливается по микробатчам, GradScaler один на шаг,
      клиппинг -- после unscale_ на полном шаге.
    - Если по ckpt_path лежит чекпоинт, обучение продолжается с него (порядок данных -- с того же шага).
    - Если итоговые веса уже есть, ничего не делается.
    - Скорость: среднее время шага по шагам 6-15 пишется в лог (step_sec); если на GPU оно больше
      cfg.speed_check.finetune_step_warn_sec, печатается предупреждение (обучение не прерывается).
    """
    sc = cfg.pretrain if stage == "pretrain" else cfg.finetune
    data_dir, ckpt_path = Path(data_dir), Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    fin = final_path(ckpt_path)
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    if fin.exists():
        rows = read_log(log_path)
        last = lambda key: next((r[key] for r in reversed(rows) if key in r), None)  # noqa: E731
        return {"steps": sc.steps, "train_loss": last("train_loss"), "val_loss": last("val_loss"),
                "step_sec": last("step_sec"), "data_wait_sec": None, "final_path": str(fin), "log_path": log_path}

    seed_everything(cfg.seeds.init)
    dev = next(model.parameters()).device
    amp_dtype = getattr(torch, cfg.precision.autocast_dtype)
    use_amp = dev.type == "cuda"

    def autocast():
        return torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()

    prefix = "pretrain" if stage == "pretrain" else "long"
    val_bin = data_dir / f"{prefix}_val.bin"
    val_every = 100 if stage == "pretrain" else 50
    micro = sc.micro_batch_size or sc.batch_size
    if sc.batch_size % micro != 0:
        raise ValueError(f"batch_size={sc.batch_size} должен делиться на micro_batch_size={micro}")
    accum = sc.batch_size // micro
    n_eval_samples = 160 if stage == "pretrain" else 200
    eval_batches = -(-n_eval_samples // micro)

    # --- старт или resume
    step, resumed = 0, None
    if ckpt_path.exists():
        resumed = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(resumed["model"])
        step = resumed["step"]
    elif init_from is not None:
        load_checkpoint(model, init_from)
    model.set_yarn_scale(sc.yarn_scale)

    opt = build_optimizer(model, kind, sc, cfg.optim)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if resumed is not None:
        opt.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
    del resumed

    rows = [r for r in read_log(log_path) if r["step"] <= step]  # после сбоя отбрасываем «хвост» лога
    if log_path is not None:
        _write_log(log_path, rows)
    last_train = next((r["train_loss"] for r in reversed(rows) if "train_loss" in r), None)
    last_val = next((r["val_loss"] for r in reversed(rows) if "val_loss" in r), None)
    step_sec = next((r["step_sec"] for r in reversed(rows) if "step_sec" in r), None)

    def log(row: dict) -> None:
        rows.append(row)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def save_ckpt() -> None:
        _atomic_save(
            {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "scaler": scaler.state_dict(),
             "kind": kind, "stage": stage},
            ckpt_path,
        )

    model.train()
    loader = make_train_loader(cfg, stage, data_dir, start_step=step, num_workers=num_workers)
    batches = iter(loader)
    t0 = time.time()
    step_times: dict[int, float] = {}
    data_wait, steps_run = 0.0, 0
    pbar = tqdm(total=sc.steps, initial=step, desc=f"{stage}/{kind}", dynamic_ncols=True)
    while step < sc.steps:
        lr = wsd_lr(step, sc.lr, sc.steps, sc.warmup_steps, sc.decay_frac, sc.final_lr_frac)
        opt.set_lr(lr)
        opt.zero_grad()
        _sync(dev)
        t_step = time.time()
        ce_sum = aux_sum = 0.0
        for _ in range(accum):
            t_data = time.time()
            xm, ym = next(batches)
            data_wait += time.time() - t_data
            with autocast():
                out = model(xm.to(dev, non_blocking=True), targets=ym.to(dev, non_blocking=True))
            scaler.scale(out["loss"] / accum).backward()
            ce_sum += out["ce_loss"].item() / accum
            aux_sum += out["aux_loss"].item() / accum
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        _sync(dev)
        step += 1
        steps_run += 1
        step_times[step] = time.time() - t_step
        pbar.update(1)

        if step == SPEED_LAST_STEP and step_sec is None:
            window = [step_times[s] for s in range(SPEED_WARMUP_STEPS + 1, SPEED_LAST_STEP + 1) if s in step_times]
            if len(window) == SPEED_LAST_STEP - SPEED_WARMUP_STEPS:
                step_sec = sum(window) / len(window)
                log({"step": step, "step_sec": step_sec})
                if dev.type == "cuda" and stage == "finetune" and step_sec > cfg.speed_check.finetune_step_warn_sec:
                    msg = (f"шаг подозрительно медленный ({step_sec:.1f} с при пороге "
                           f"{cfg.speed_check.finetune_step_warn_sec} с): проверьте, что внимание считается через "
                           "F.scaled_dot_product_attention с is_causal=True при обучении (приложение A.2)")
                    warnings.warn(msg)
                    print("ПРЕДУПРЕЖДЕНИЕ:", msg)

        dense = stage == "finetune" and step <= FINETUNE_DENSE_LOG_STEPS
        is_last = step == sc.steps
        if dense or step % LOG_EVERY == 0 or is_last:
            last_train = ce_sum
            log({"step": step, "train_loss": ce_sum, "aux_loss": aux_sum, "lr": lr, "grad_norm": grad_norm,
                 "elapsed": time.time() - t0})
            pbar.set_postfix(loss=f"{ce_sum:.3f}", lr=f"{lr:.1e}")
        if step % val_every == 0 or is_last:
            last_val = eval_loss(model, val_bin, sc.seq_len, eval_batches, batch_size=micro)
            load = [layer.moe.last_load.tolist() for layer in model.layers]
            log({"step": step, "val_loss": last_val, "moe_load": load})
            model.train()
        if step % sc.ckpt_every == 0 or is_last:
            save_ckpt()
    pbar.close()
    del batches, loader  # останавливаем воркеры DataLoader

    if not ckpt_path.exists():  # resume уже на последнем шаге, а чекпоинт пропал
        save_ckpt()
    save_checkpoint(model, fin, checkpoint_meta(kind, step, step * sc.batch_size * sc.seq_len))
    return {"steps": step, "train_loss": last_train, "val_loss": last_val, "step_sec": step_sec,
            "data_wait_sec": data_wait / max(steps_run, 1), "final_path": str(fin), "log_path": log_path}
