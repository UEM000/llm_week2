from functools import partial

import numpy as np
import torch

IGNORE_INDEX = -100


def collate_lm(batch, kind: str, ignore_index: int = IGNORE_INDEX):
    """Складывает элементы в [B, L] (int64) и строит (input_ids, targets).

    kind="pretrain": элементы длины L + 1, input_ids = x[:, :-1], targets = x[:, 1:].
    kind="long":     элементы длины ровно L; input_ids = x, targets = x со сдвигом влево на 1, а последняя
                     позиция = ignore_index (длина входа остаётся L, и бакеты позиций считаются на тех же L позициях).
    Паддинга нет: все элементы батча одной длины.
    """
    lengths = {len(x) for x in batch}
    assert len(lengths) == 1, f"в батче элементы разной длины: {sorted(lengths)}"
    x = torch.from_numpy(np.stack(batch).astype(np.int64))
    if kind == "pretrain":
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()
    if kind == "long":
        targets = torch.full_like(x, ignore_index)
        targets[:, :-1] = x[:, 1:]
        return x, targets
    raise ValueError(f"неизвестный kind: {kind!r}")


def make_collate(kind: str):
    """Фабрика коллаторов для DataLoader (partial над функцией верхнего уровня -- пиклится в воркеры)."""
    return partial(collate_lm, kind=kind)


def collate_passkey(batch):
    """Батч passkey-примеров одной длины: input_ids [B, L], answer_ids [B, A] и метаданные."""
    assert len({b["length"] for b in batch}) == 1, "passkey-батч должен состоять из примеров одной длины"
    return {
        "input_ids": torch.from_numpy(np.stack([b["input_ids"] for b in batch])),
        "answer_ids": torch.from_numpy(np.stack([b["answer_ids"] for b in batch])),
        "length": batch[0]["length"],
        "depth": [b["depth"] for b in batch],
    }
