"""Совместимость с выданными чекпоинтами: имена и формы параметров модели нельзя менять (офлайн, без скачивания)."""
import json
from pathlib import Path

import pytest
import torch

from kestrel.model import build_model
from kestrel.model.model import load_checkpoint, save_checkpoint

SPEC = Path(__file__).parent / "fixtures" / "state_dict_spec.json"


def test_full_model_matches_state_dict_spec(full_cfg):
    spec = [(name, tuple(shape)) for name, shape in json.loads(SPEC.read_text(encoding="utf-8"))]
    got = [(name, tuple(t.shape)) for name, t in build_model(full_cfg, device="meta").state_dict().items()]
    assert [n for n, _ in got] == [n for n, _ in spec], "имена параметров изменились (не переименовывайте атрибуты)"
    assert got == spec


def test_tiny_state_dict_loads_strictly(tiny_cfg, golden):
    model = build_model(tiny_cfg)
    model.load_state_dict(golden("tiny_model")["state_dict"], strict=True)


def test_load_checkpoint_roundtrip_and_errors(tiny_cfg, tmp_path):
    model = build_model(tiny_cfg)
    save_checkpoint(model, tmp_path / "c.pt", {"optimizer": "adamw", "steps": 3})
    other = build_model(tiny_cfg)
    meta = load_checkpoint(other, tmp_path / "c.pt")
    assert meta == {"optimizer": "adamw", "steps": 3}
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(model.state_dict().values(), other.state_dict().values()))  # хранится fp32

    bad = {k.replace("q_proj", "q_projection"): v for k, v in model.state_dict().items()}
    torch.save({"model": bad, "meta": {}}, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError, match="missing"):
        load_checkpoint(build_model(tiny_cfg), tmp_path / "bad.pt")

    torch.save(model.state_dict(), tmp_path / "raw.pt")  # чистый state_dict тоже принимается
    assert load_checkpoint(build_model(tiny_cfg), tmp_path / "raw.pt") == {}
