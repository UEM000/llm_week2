"""Пайплайн данных: датасеты, коллатор, сэмплер, DataLoader (CPU, на маленьких синтетических файлах)."""
import pickle

import numpy as np
import pytest
import torch

from kestrel.config import smoke_config
from kestrel.data.collate import IGNORE_INDEX, collate_lm
from kestrel.data.datasets import LongSequences, PasskeyEval, PretrainWindows
from kestrel.data.loader import make_eval_loader, make_passkey_loader, make_train_loader
from kestrel.data.sampler import ResumableRandomSampler
from kestrel.data.synthetic import make_synthetic_dataset


@pytest.fixture
def cfg():
    return smoke_config()  # finetune: seq 128, batch 4 = micro 2 x accum 2; pretrain: seq 32, batch 8


@pytest.fixture
def data(tmp_path, cfg):
    return make_synthetic_dataset(tmp_path / "data", cfg, n_pretrain_windows=50, n_long=24, n_val=4)


def take(loader, n):
    it = iter(loader)
    return [next(it) for _ in range(n)]


def test_datasets_shapes_and_lazy_memmap(tmp_path):
    np.arange(5 * 8 + 1, dtype=np.uint16).tofile(tmp_path / "pretrain_x.bin")
    np.arange(6 * 8, dtype=np.uint16).tofile(tmp_path / "long_x.bin")
    pre, long = PretrainWindows(tmp_path / "pretrain_x.bin", 8), LongSequences(tmp_path / "long_x.bin", 8)
    assert len(pre) == 5 and len(long) == 6
    assert pre._mm is None and long._mm is None  # memmap открывается лениво
    assert pre[1].shape == (9,) and pre[1].dtype == np.uint16 and pre[1][0] == 8
    assert long[2].shape == (8,) and long[2][0] == 16
    assert pickle.loads(pickle.dumps(pre))._mm is None
    with pytest.raises(IndexError):
        pre[5]


def test_collate_pretrain_and_long():
    x = [np.arange(9, dtype=np.uint16) + 10 * i for i in range(3)]
    ids, tgt = collate_lm(x, "pretrain")
    assert ids.dtype == tgt.dtype == torch.int64 and ids.shape == tgt.shape == (3, 8)
    assert torch.equal(ids[:, 1:], tgt[:, :-1])
    ids, tgt = collate_lm([a[:8] for a in x], "long")
    assert ids.shape == tgt.shape == (3, 8)
    assert torch.equal(tgt[:, :-1], ids[:, 1:]) and (tgt[:, -1] == IGNORE_INDEX).all()
    with pytest.raises(AssertionError):
        collate_lm([x[0], x[1][:5]], "pretrain")


def test_loader_shapes(cfg, data):
    ids, tgt = take(make_train_loader(cfg, "finetune", data, num_workers=0), 1)[0]
    assert ids.shape == (2, 128) and ids.dtype == torch.int64
    assert (tgt[:, -1] == IGNORE_INDEX).all() and torch.equal(tgt[:, :-1], ids[:, 1:])
    ids, tgt = take(make_train_loader(cfg, "pretrain", data, num_workers=0), 1)[0]
    assert ids.shape == (8, 32) and torch.equal(tgt[:, :-1], ids[:, 1:])


def test_same_seed_same_batches_other_seed_differs(cfg, data):
    a = take(make_train_loader(cfg, "pretrain", data, num_workers=0), 4)
    b = take(make_train_loader(cfg, "pretrain", data, num_workers=0), 4)
    assert all(torch.equal(x[0], y[0]) for x, y in zip(a, b))
    cfg.seeds.data_order += 1
    c = take(make_train_loader(cfg, "pretrain", data, num_workers=0), 4)
    assert not all(torch.equal(x[0], y[0]) for x, y in zip(a, c))


def test_sampler_is_permutation_and_resumes_across_epochs():
    n = 7
    it = iter(ResumableRandomSampler(n, seed=3))
    stream = [next(it) for _ in range(4 * n)]
    for e in range(3):
        assert sorted(stream[e * n : (e + 1) * n]) == list(range(n))
    assert stream[:n] != stream[n : 2 * n]  # у каждой эпохи свой порядок
    for start in (0, 4, n, n + 3, 2 * n + 1):
        it2 = iter(ResumableRandomSampler(n, seed=3, start_index=start))
        assert [next(it2) for _ in range(n)] == stream[start : start + n]


def test_resume_gives_same_batches_as_continuous_run(cfg, data):
    # шаг = grad_accum (2) микробатча по micro_batch_size (2) = глобальный батч 4
    accum = cfg.finetune.grad_accum
    full = take(make_train_loader(cfg, "finetune", data, num_workers=0), 6 * accum)
    for k in (1, 3, 5):
        resumed = take(make_train_loader(cfg, "finetune", data, start_step=k, num_workers=0), (6 - k) * accum)
        assert all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(full[k * accum :], resumed))


def test_workers_give_same_batches_as_no_workers(cfg, data):
    a = take(make_train_loader(cfg, "finetune", data, num_workers=0), 6)
    b = take(make_train_loader(cfg, "finetune", data, num_workers=2), 6)
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))


def test_eval_loader_is_sequential(cfg, data):
    ds = LongSequences(data / "long_val.bin", cfg.finetune.seq_len)
    batches = list(make_eval_loader(data / "long_val.bin", cfg.finetune.seq_len, 2, n_samples=4))
    assert len(batches) == 2
    assert torch.equal(batches[0][0][0], torch.from_numpy(ds[0].astype(np.int64)))
    assert torch.equal(batches[1][0][1], torch.from_numpy(ds[3].astype(np.int64)))


def test_passkey_batches_have_single_length(cfg, data):
    path = data / "passkey_eval.jsonl"
    ds = PasskeyEval(path)
    assert sorted(set(ds.lengths)) == [cfg.finetune.seq_len // 2, cfg.finetune.seq_len]
    seen = 0
    for batch in make_passkey_loader(path, batch_size=7, seed=1):
        L = batch["length"]
        assert batch["input_ids"].shape[1] + batch["answer_ids"].shape[1] == L
        seen += batch["input_ids"].shape[0]
    assert seen == len(ds)
    n = sum(b["input_ids"].shape[0] for b in make_passkey_loader(path, 7, seed=1, n_per_length=12))
    assert n == 24
