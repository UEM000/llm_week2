from contextlib import nullcontext

import torch
import torch.nn.functional as F

from kestrel.data.collate import IGNORE_INDEX
from kestrel.data.loader import make_eval_loader, make_passkey_loader


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _autocast(model):
    """fp16-autocast на GPU, обычный fp32 на CPU."""
    if _device(model).type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def eval_loss(model, path, seq_len: int, n_batches: int | None = None, batch_size: int = 4) -> float:
    """Средний кросс-энтропийный лосс (без aux) по первым n_batches*batch_size элементам файла (None -- по всем).

    Позиции с целью IGNORE_INDEX не учитываются.
    """
    was_training = model.training
    model.eval()
    dev = _device(model)
    n_samples = None if n_batches is None else n_batches * batch_size
    total, count = 0.0, 0
    for x, y in make_eval_loader(path, seq_len, batch_size, n_samples):
        with _autocast(model):
            logits = model(x.to(dev))["logits"]
        y = y.to(dev)
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                 ignore_index=IGNORE_INDEX, reduction="sum").item()
        count += (y != IGNORE_INDEX).sum().item()
    model.train(was_training)
    return total / count


def checkpoint_val_loss(model, data_dir, seq_len: int = 512, batch_size: int = 32) -> float:
    """Val loss выданных чекпоинтов: весь pretrain_val.bin при длине 512 (на GPU -- в fp16-autocast).

    Один и тот же протокол используется преподавателем (reference_val_loss) и студенческим ноутбуком.
    """
    return eval_loss(model, f"{data_dir}/pretrain_val.bin", seq_len, None, batch_size)


@torch.no_grad()
def loss_by_position_buckets(model, path, seq_len: int = 2048, bucket: int = 512, batch_size: int = 4) -> dict:
    """Средний лосс по корзинам позиций внутри куска: {'0-511': ..., '512-1023': ..., ...} (по всем элементам файла).

    Позиции с целью IGNORE_INDEX (последняя позиция длинных последовательностей) не учитываются.
    """
    was_training = model.training
    model.eval()
    dev = _device(model)
    n_buckets = seq_len // bucket
    sums = torch.zeros(n_buckets, dtype=torch.float64)
    counts = torch.zeros(n_buckets, dtype=torch.float64)
    for x, y in make_eval_loader(path, seq_len, batch_size):
        with _autocast(model):
            logits = model(x.to(dev))["logits"]
        y = y.to(dev)
        per_tok = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                  ignore_index=IGNORE_INDEX, reduction="none").reshape(y.shape)
        valid = y != IGNORE_INDEX
        for b in range(n_buckets):
            sl = slice(b * bucket, (b + 1) * bucket)
            sums[b] += per_tok[:, sl].sum().item()
            counts[b] += valid[:, sl].sum().item()
    model.train(was_training)
    means = (sums / counts).tolist()
    return {f"{b * bucket}-{(b + 1) * bucket - 1}": means[b] for b in range(n_buckets)}


@torch.no_grad()
def passkey_accuracy(model, jsonl_path, seed: int, n_per_length: int = 250, batch_size: int = 8) -> dict:
    """Точность поиска passkey: доля примеров, где ВСЕ цифры ответа предсказаны верно (teacher forcing).

    Для каждой длины из файла берётся до n_per_length примеров (по умолчанию -- все: 5 глубин x 50 ключей;
    выбор -- numpy.random.Generator(seed)). Модель получает prompt + answer[:-1] и должна по argmax угадать
    каждую цифру ответа. Возвращает {'<length>': accuracy, ..., 'all': accuracy}.
    """
    was_training = model.training
    model.eval()
    dev = _device(model)
    hits, totals = {}, {}
    for batch in make_passkey_loader(jsonl_path, batch_size, seed, n_per_length):
        answer = batch["answer_ids"].to(dev)
        ids = torch.cat([batch["input_ids"].to(dev), answer[:, :-1]], dim=1)
        with _autocast(model):
            logits = model(ids)["logits"]
        pred = logits[:, -answer.shape[1]:].argmax(-1)
        key = str(batch["length"])
        hits[key] = hits.get(key, 0) + (pred == answer).all(dim=1).sum().item()
        totals[key] = totals.get(key, 0) + answer.shape[0]
    model.train(was_training)
    result = {k: hits[k] / totals[k] for k in sorted(hits, key=int)}
    result["all"] = sum(hits.values()) / sum(totals.values())
    return result
