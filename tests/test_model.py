import pytest
import torch

from kestrel.model import build_model, count_parameters
from kestrel.optim.build import build_optimizer
from kestrel.seed import make_generator, seed_everything


def test_parameter_count_full_config(full_cfg):
    model = build_model(full_cfg, device="meta")
    assert count_parameters(model) == 22_303_488


def test_forward_outputs(tiny_cfg, gen):
    model = build_model(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.model.vocab_size, (2, 11), generator=gen)
    out = model(ids)
    assert out["logits"].shape == (2, 11, tiny_cfg.model.vocab_size)
    assert out["aux_loss"].ndim == 0 and "loss" not in out
    out = model(ids, targets=ids)
    assert {"logits", "aux_loss", "loss"} <= set(out)
    ce = torch.nn.functional.cross_entropy(out["logits"].reshape(-1, out["logits"].shape[-1]), ids.reshape(-1))
    assert torch.allclose(out["loss"], ce + out["aux_loss"], atol=1e-10)
    # при случайной инициализации aux каждого слоя ~ alpha, суммируется по слоям
    assert 0.5 * tiny_cfg.moe.aux_alpha * tiny_cfg.model.n_layers < out["aux_loss"].item() < 3 * (
        tiny_cfg.moe.aux_alpha * tiny_cfg.model.n_layers)


def test_init_scheme(tiny_cfg):
    model = build_model(tiny_cfg)
    std = tiny_cfg.model.init_std
    assert abs(model.embed.weight.std().item() - std) < 0.2 * std
    deep = std / (2 * tiny_cfg.model.n_layers) ** 0.5
    assert abs(model.layers[0].attn.o_proj.weight.std().item() - deep) < 0.25 * deep
    assert abs(model.layers[0].moe.experts[0].w_down.weight.std().item() - deep) < 0.25 * deep
    assert torch.equal(model.final_norm.weight, torch.ones_like(model.final_norm.weight))
    assert model.lm_head.weight is not model.embed.weight


def test_generate_with_cache_equals_without(tiny_cfg, gen):
    model = build_model(tiny_cfg)
    prompt = torch.randint(0, tiny_cfg.model.vocab_size, (2, 5), generator=gen)
    with_cache = model.generate(prompt, 8, use_cache=True)
    without_cache = model.generate(prompt, 8, use_cache=False)
    assert with_cache.shape == (2, 13)
    assert torch.equal(with_cache, without_cache)


def test_yarn_scale_changes_outputs(tiny_cfg, gen):
    model = build_model(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.model.vocab_size, (1, 12), generator=gen)
    base = model(ids)["logits"]
    model.set_yarn_scale(4.0)
    scaled = model(ids)["logits"]
    assert not torch.allclose(base, scaled, atol=1e-6)
    model.set_yarn_scale(1.0)
    assert torch.allclose(model(ids)["logits"], base, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("kind,lr,threshold", [("adamw", 3e-3, 0.1), ("muon", 3e-3, 0.1)])
def test_overfit_single_batch(tiny_cfg, kind, lr, threshold):
    """Модель + оптимизатор способны выучить один батч наизусть."""
    seed_everything(tiny_cfg.seeds.tests)
    gen = make_generator(tiny_cfg.seeds.tests)
    model = build_model(tiny_cfg)
    ids = torch.randint(0, tiny_cfg.model.vocab_size, (4, 25), generator=gen)
    x, y = ids[:, :-1], ids[:, 1:]
    stage = tiny_cfg.pretrain
    stage.lr = lr
    opt = build_optimizer(model, kind, stage, tiny_cfg.optim)
    first = None
    for _ in range(200):
        opt.zero_grad()
        out = model(x, targets=y)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tiny_cfg.optim.grad_clip)
        opt.step()
        first = first if first is not None else out["ce_loss"].item()
    final = model(x, targets=y)["ce_loss"].item()
    assert final < threshold, f"{kind}: loss {first:.3f} -> {final:.3f}"
