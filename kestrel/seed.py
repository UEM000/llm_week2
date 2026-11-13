"""Управление случайностью."""
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Фиксирует random, numpy и torch (CPU и CUDA), отключает недетерминизм cudnn."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device="cpu") -> torch.Generator:
    """Локальный генератор: через него создаются случайные входы в тестах."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g
