"""Датасеты поверх .bin (uint16, memmap) и jsonl. memmap открывается лениво, внутри процесса-воркера."""
import json
import os

import numpy as np
from torch.utils.data import Dataset


class _MemmapDataset(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self._mm = None  # открываем при первом обращении: так датасет безопасно пиклится в воркеры DataLoader
        self.n_tokens = int(np.memmap(self.path, dtype=np.uint16, mode="r").shape[0])

    @property
    def mm(self) -> np.ndarray:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint16, mode="r")
        return self._mm

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state


class PretrainWindows(_MemmapDataset):
    """Плоский поток токенов. Элемент i -- окно из seq_len + 1 токенов, начиная с i * seq_len
    (соседние окна перекрываются на один токен, чтобы у каждой позиции была цель)."""

    kind = "pretrain"

    def __init__(self, path, seq_len: int = 512):
        super().__init__(path)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (self.n_tokens - 1) // self.seq_len

    def __getitem__(self, i: int) -> np.ndarray:
        if not 0 <= i < len(self):
            raise IndexError(i)
        start = i * self.seq_len
        return np.array(self.mm[start : start + self.seq_len + 1])


class LongSequences(_MemmapDataset):
    """Массив [n, seq_len] uint16: каждая строка -- один связный фрагмент документа ровно в seq_len токенов."""

    kind = "long"

    def __init__(self, path, seq_len: int = 2048):
        super().__init__(path)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.n_tokens // self.seq_len

    def __getitem__(self, i: int) -> np.ndarray:
        if not 0 <= i < len(self):
            raise IndexError(i)
        start = i * self.seq_len
        return np.array(self.mm[start : start + self.seq_len])


class PasskeyEval(Dataset):
    """Примеры passkey из jsonl: {'input_ids', 'answer_ids', 'length', 'depth'}."""

    def __init__(self, jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.lengths = [r["length"] for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {"input_ids": np.asarray(r["input_ids"], dtype=np.int64),
                "answer_ids": np.asarray(r["answer_ids"], dtype=np.int64),
                "length": r["length"], "depth": r["depth"]}


def make_dataset(path, seq_len: int, kind: str | None = None) -> Dataset:
    """pretrain_*.bin -> PretrainWindows, long_*.bin -> LongSequences (вид можно указать явно)."""
    kind = kind or ("pretrain" if os.path.basename(str(path)).startswith("pretrain") else "long")
    return (PretrainWindows if kind == "pretrain" else LongSequences)(path, seq_len)
