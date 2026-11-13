from pathlib import Path

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from kestrel.config import load_config, tiny_config
from kestrel.seed import make_generator, seed_everything

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _deterministic_fp64():
    """Каждый тест: сид из конфига, float64 по умолчанию и math-бэкенд SDPA.

    Допуски рассчитаны на fp64. Быстрые CPU-ядра SDPA считают с точностью порядка 1e-7 даже для float64,
    поэтому в тестах фиксируется math-бэкенд (на обучении на GPU выбор бэкенда остаётся за PyTorch).
    """
    seed_everything(load_config().seeds.tests)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    with sdpa_kernel(SDPBackend.MATH):
        yield
    torch.set_default_dtype(prev)


@pytest.fixture
def full_cfg():
    return load_config()


@pytest.fixture
def tiny_cfg():
    return tiny_config()


@pytest.fixture
def gen():
    """Локальный генератор: случайные входы в тестах создаются только через него."""
    return make_generator(load_config().seeds.tests)


@pytest.fixture
def golden():
    """golden('attention') -> dict из tests/fixtures/attention_golden.pt."""

    def _load(name: str) -> dict:
        return torch.load(FIXTURES / f"{name}_golden.pt", map_location="cpu", weights_only=True)

    return _load
