"""Проверки готовых модулей (RoPE, норма, кэш, Muon, расписание, данные, метрики, trainer)."""
import pytest
import torch

from kestrel.config import smoke_config
from kestrel.data.synthetic import make_synthetic_dataset
from kestrel.model import KVCache, build_model
from kestrel.model.model import load_checkpoint
from kestrel.model.norm import RMSNorm
from kestrel.model.rope import RotaryEmbedding, apply_rope, rope_inv_freq
from kestrel.optim.muon import Muon, newton_schulz
from kestrel.optim.schedule import wsd_lr
from kestrel.train.evaluate import checkpoint_val_loss, eval_loss, loss_by_position_buckets, passkey_accuracy
from kestrel.train.metrics import expert_load, relative_drift, stable_rank


def test_rmsnorm():
    x = torch.randn(3, 5, 8)
    norm = RMSNorm(8, 1e-6)
    y = norm(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(3, 5), atol=1e-4)
    assert RMSNorm(8, 1e-6)(x.half().float().half()).dtype == torch.float16


def test_rope_properties():
    inv = rope_inv_freq(16, 10000.0)
    assert inv.shape == (8,) and inv.dtype == torch.float32 and inv[0] == 1.0
    x = torch.randn(1, 2, 6, 16)
    out = apply_rope(x, torch.arange(6), inv)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)
    # скалярное произведение зависит только от разности позиций
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    dots = []
    for p, s in [(3, 1), (10, 8), (102, 100)]:
        qq = apply_rope(q, torch.tensor([p]), inv)
        kk = apply_rope(k, torch.tensor([s]), inv)
        dots.append((qq * kk).sum().item())
    assert max(dots) - min(dots) < 1e-3


def test_rotary_set_scale_updates_frequencies():
    rot = RotaryEmbedding(32, 10000.0, 512, 1.0, 32.0)
    base = rot.inv_freq.clone()
    assert rot.mscale == 1.0
    rot.set_scale(4.0)
    assert not torch.equal(rot.inv_freq, base) and abs(rot.mscale - 1.138629) < 1e-5
    rot.set_scale(1.0)
    assert torch.equal(rot.inv_freq, base)


def test_kv_cache():
    cache = KVCache(2)
    assert cache.seq_len(0) == 0
    k, v = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k2, _ = cache.update(0, k, v)
    assert k2.shape[2] == 3 and cache.seq_len(0) == 3 and cache.seq_len(1) == 0
    k3, _ = cache.update(0, k[:, :, :1], v[:, :, :1])
    assert k3.shape[2] == 4
    cache.reset()
    assert cache.seq_len(0) == 0


def test_newton_schulz_orthogonalizes():
    g = torch.randn(32, 64)
    s = torch.linalg.svdvals(newton_schulz(g, 5))
    assert s.min() > 0.4 and s.max() < 1.3
    assert newton_schulz(g.T.contiguous(), 5).shape == (64, 32)


def test_muon_step_moves_2d_params_and_rejects_vectors():
    w = torch.nn.Parameter(torch.randn(8, 16))
    w.grad = torch.randn(8, 16)
    before = w.detach().clone()
    Muon([w], lr=0.01).step()
    assert not torch.equal(w.detach(), before)
    v = torch.nn.Parameter(torch.randn(8))
    v.grad = torch.randn(8)
    with pytest.raises(ValueError):
        Muon([v], lr=0.01).step()


def test_wsd_schedule():
    kw = dict(peak=1.0, total_steps=100, warmup_steps=10, decay_frac=0.2, final_lr_frac=0.1)
    assert wsd_lr(0, **kw) == pytest.approx(0.1)
    assert wsd_lr(9, **kw) == pytest.approx(1.0)
    assert wsd_lr(50, **kw) == 1.0
    assert wsd_lr(79, **kw) == 1.0
    assert wsd_lr(99, **kw) == pytest.approx(0.1)
    lrs = [wsd_lr(s, **kw) for s in range(80, 100)]
    assert all(a > b for a, b in zip(lrs, lrs[1:]))


def test_metrics():
    assert stable_rank(torch.eye(6)) == pytest.approx(6.0)
    rank1 = torch.ones(5, 5)
    assert stable_rank(rank1) == pytest.approx(1.0)
    w0 = torch.randn(4, 4)
    assert relative_drift(w0, w0) == 0.0
    assert relative_drift(2 * w0, w0) == pytest.approx(1.0, rel=1e-5)


def test_synthetic_dataset_and_pipeline(tmp_path, monkeypatch):
    """Сквозная проверка: данные -> trainer (resume после сбоя даёт те же веса) -> оценка.

    Использует студенческие AdamW и split_params (через build_optimizer).
    """
    import kestrel.train.trainer as trainer

    cfg = smoke_config()
    cfg.pretrain.steps = 6
    cfg.pretrain.ckpt_every = 3
    data = make_synthetic_dataset(tmp_path / "data", cfg, n_pretrain_windows=60, n_long=20, n_val=4)
    assert (data / "passkey_eval.jsonl").exists()

    def fresh():
        torch.manual_seed(0)
        return build_model(cfg)

    run = lambda model, ck, log=None: trainer.train(  # noqa: E731
        model, "muon", "pretrain", cfg, data, ck, log_path=log, num_workers=0)

    straight = fresh()
    ck_a = tmp_path / "a" / "ck.pt"
    out = run(straight, ck_a, str(tmp_path / "a.jsonl"))
    assert trainer.is_done(ck_a) and out["val_loss"] is not None
    meta = load_checkpoint(fresh(), trainer.final_path(ck_a))
    assert meta["optimizer"] == "muon" and meta["steps"] == 6 and meta["tokens"] == 6 * 8 * 32
    assert run(fresh(), ck_a)["val_loss"] is None  # уже готово: ничего не делает

    # сбой сразу после чекпоинта на шаге 3, затем resume
    class Crash(Exception):
        pass

    real_save = trainer._atomic_save

    def save_then_crash(obj, path):
        real_save(obj, path)
        if isinstance(obj, dict) and obj.get("step") == 3:
            raise Crash

    ck_b = tmp_path / "b" / "ck.pt"
    monkeypatch.setattr(trainer, "_atomic_save", save_then_crash)
    with pytest.raises(Crash):
        run(fresh(), ck_b, str(tmp_path / "b.jsonl"))
    monkeypatch.setattr(trainer, "_atomic_save", real_save)
    resumed = fresh()
    run(resumed, ck_b, str(tmp_path / "b.jsonl"))
    for (name, p1), (_, p2) in zip(straight.named_parameters(), resumed.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-9), name
    steps = lambda path: [r["step"] for r in trainer.read_log(path)]  # noqa: E731
    assert steps(tmp_path / "a.jsonl") == steps(tmp_path / "b.jsonl")

    # оценка
    assert eval_loss(straight, data / "pretrain_val.bin", cfg.pretrain.seq_len, 2) > 0
    assert checkpoint_val_loss(straight, data, cfg.pretrain.seq_len, 4) > 0
    buckets = loss_by_position_buckets(straight, data / "long_val.bin", cfg.finetune.seq_len, 32)
    assert len(buckets) == cfg.finetune.seq_len // 32
    acc = passkey_accuracy(straight, data / "passkey_eval.jsonl", cfg.seeds.passkey_eval, n_per_length=5)
    assert 0.0 <= acc["all"] <= 1.0 and str(cfg.finetune.seq_len) in acc
    load = expert_load(straight, data / "pretrain_val.bin", cfg.pretrain.seq_len, n_batches=2, batch_size=4)
    assert len(load) == cfg.model.n_layers and abs(sum(load[0]) - 1.0) < 1e-5

    # дообучение из чекпоинта: микробатчи + замер скорости пишутся в лог
    cfg.finetune.steps = 16
    cfg.finetune.ckpt_every = 8
    ft = trainer.train(fresh(), "adamw", "finetune", cfg, data, tmp_path / "ft" / "ck.pt",
                       init_from=str(trainer.final_path(ck_a)), log_path=str(tmp_path / "ft.jsonl"), num_workers=0)
    assert ft["step_sec"] is not None and ft["val_loss"] is not None
