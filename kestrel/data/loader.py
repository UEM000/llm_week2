"""DataLoader'ы для обучения и оценки: Dataset + детерминированный сэмплер + коллатор (torch.utils.data)."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from kestrel.data.collate import collate_passkey, make_collate
from kestrel.data.datasets import PasskeyEval, make_dataset
from kestrel.data.sampler import LengthGroupedBatchSampler, ResumableRandomSampler


class _WorkerInit:
    """Сидирует numpy/random в воркерах (класс, а не замыкание: пиклится на Windows и macOS)."""

    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        np.random.seed(self.seed + worker_id)
        random.seed(self.seed + worker_id)


def make_train_loader(cfg, stage: str, data_dir, start_step: int = 0, num_workers: int = 2) -> DataLoader:
    """Бесконечный загрузчик микробатчей для стадии 'pretrain' или 'finetune'.

    Размер батча DataLoader = micro_batch_size; один оптимизаторный шаг = grad_accum микробатчей.
    start_step -- с какого шага продолжать: порядок пропускается по индексам, данные не читаются.
    """
    sc = cfg.pretrain if stage == "pretrain" else cfg.finetune
    prefix = "pretrain" if stage == "pretrain" else "long"
    ds = make_dataset(f"{data_dir}/{prefix}_train.bin", sc.seq_len, kind=prefix)
    micro = sc.micro_batch_size or sc.batch_size
    sampler = ResumableRandomSampler(len(ds), cfg.seeds.data_order, start_index=start_step * sc.batch_size)
    kwargs = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=_WorkerInit(cfg.seeds.data_order))
    return DataLoader(ds, batch_size=micro, sampler=sampler, drop_last=True, collate_fn=make_collate(prefix),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), **kwargs)


def make_eval_loader(path, seq_len: int, batch_size: int, n_samples: int | None = None,
                     kind: str | None = None) -> DataLoader:
    """Последовательный (без перемешивания) загрузчик для оценки; n_samples -- сколько первых элементов взять."""
    ds = make_dataset(path, seq_len, kind)
    kind = ds.kind
    if n_samples is not None:
        ds = Subset(ds, range(min(n_samples, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate(kind), num_workers=0)


def make_passkey_loader(jsonl_path, batch_size: int = 8, seed: int = 0, n_per_length: int | None = None) -> DataLoader:
    """Батчи passkey-примеров, сгруппированные по длине."""
    ds = PasskeyEval(jsonl_path)
    sampler = LengthGroupedBatchSampler(ds.lengths, batch_size, seed, n_per_length)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_passkey, num_workers=0)
