import numpy as np
import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler):
    """Бесконечный детерминированный сэмплер: эпоха e -- перестановка от torch.Generator с сидом seed + e.

    start_index пропускает уже выданные индексы НА УРОВНЕ ИНДЕКСОВ, без чтения данных: resume с шага k
    мгновенный и даёт ровно те же батчи, что непрерывный прогон (start_index = k * глобальный_батч).
    """

    def __init__(self, n: int, seed: int, start_index: int = 0):
        self.n, self.seed, self.start_index = n, seed, start_index

    def _perm(self, epoch: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        return torch.randperm(self.n, generator=g)

    def __iter__(self):
        epoch, skip = divmod(self.start_index, self.n)
        while True:
            perm = self._perm(epoch)
            yield from perm[skip:].tolist()
            epoch, skip = epoch + 1, 0


class LengthGroupedBatchSampler:
    """Для passkey: батчи из индексов с одинаковым length. n_per_length -- сколько примеров каждой длины брать
    (случайный выбор numpy-генератором с сидом seed; None -- все)."""

    def __init__(self, lengths, batch_size: int, seed: int = 0, n_per_length: int | None = None):
        rng = np.random.default_rng(seed)
        self.batches = []
        for length in sorted(set(lengths)):
            idx = np.array([i for i, l in enumerate(lengths) if l == length])
            if n_per_length is not None and n_per_length < len(idx):
                idx = rng.choice(idx, size=n_per_length, replace=False)
            for s in range(0, len(idx), batch_size):
                self.batches.append([int(i) for i in idx[s : s + batch_size]])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)
