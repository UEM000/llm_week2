import pytest
import torch
from torch import nn

from kestrel.model import build_model
from kestrel.optim.param_groups import split_params


def expected_names(n_layers, n_experts):
    matrix, vector_decay, no_decay = set(), {"lm_head.weight"}, {"embed.weight", "final_norm.weight"}
    for i in range(n_layers):
        p = f"layers.{i}."
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            matrix.add(f"{p}attn.{name}.weight")
        for e in range(n_experts):
            for name in ("w_gate", "w_up", "w_down"):
                matrix.add(f"{p}moe.experts.{e}.{name}.weight")
        vector_decay.add(f"{p}moe.router.weight")
        no_decay |= {f"{p}attn_norm.weight", f"{p}moe_norm.weight", f"{p}attn.q_norm.weight", f"{p}attn.k_norm.weight"}
    return matrix, vector_decay, no_decay


def names_of(model, params):
    by_id = {id(p): n for n, p in model.named_parameters()}
    return {by_id[id(p)] for p in params}


def test_groups_partition_all_parameters(tiny_cfg):
    model = build_model(tiny_cfg)
    groups = split_params(model)
    assert set(groups) == {"matrix", "vector_decay", "no_decay"}
    ids = [id(p) for g in groups.values() for p in g]
    assert len(ids) == len(set(ids)), "группы пересекаются"
    assert set(ids) == {id(p) for p in model.parameters()}, "группы не покрывают все параметры"


def test_exact_composition(tiny_cfg):
    model = build_model(tiny_cfg)
    groups = split_params(model)
    matrix, vector_decay, no_decay = expected_names(tiny_cfg.model.n_layers, tiny_cfg.model.n_experts)
    assert names_of(model, groups["matrix"]) == matrix
    assert names_of(model, groups["vector_decay"]) == vector_decay
    assert names_of(model, groups["no_decay"]) == no_decay


def test_router_norms_and_embeddings(tiny_cfg):
    model = build_model(tiny_cfg)
    groups = split_params(model)
    assert any(p is model.layers[0].moe.router.weight for p in groups["vector_decay"])
    assert not any(p is model.layers[0].moe.router.weight for p in groups["matrix"])
    assert any(p is model.embed.weight for p in groups["no_decay"])
    assert any(p is model.layers[1].attn.q_norm.weight for p in groups["no_decay"])
    assert any(p is model.final_norm.weight for p in groups["no_decay"])
    assert any(p is model.lm_head.weight for p in groups["vector_decay"])
    assert all(p.ndim == 2 for p in groups["matrix"])


def test_unassigned_parameter_raises(tiny_cfg):
    model = build_model(tiny_cfg)
    model.stray = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError):
        split_params(model)
