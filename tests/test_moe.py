import torch

from kestrel.model.moe import MoE, route, switch_aux_loss

ATOL = 1e-10


def build_moe(cfg):
    m = cfg.model
    return MoE(m.d_model, m.d_ff, m.n_experts, m.top_k, cfg.moe.aux_alpha)


def cyclic_logits(n_tokens, n_experts):
    """Токен t выбирает пару экспертов (t, t+1) mod E: нагрузка и средняя вероятность равномерны, ничьих нет."""
    logits = torch.zeros(n_tokens, n_experts, dtype=torch.float64)
    for t in range(n_tokens):
        logits[t, t % n_experts] = 2.0
        logits[t, (t + 1) % n_experts] = 1.0
    return logits


def test_shapes(tiny_cfg, gen):
    moe = build_moe(tiny_cfg)
    x = torch.randn(2, 5, tiny_cfg.model.d_model, generator=gen, dtype=torch.float64)
    y, aux = moe(x)
    assert y.shape == x.shape and aux.ndim == 0
    assert moe.router.weight.shape == (tiny_cfg.model.n_experts, tiny_cfg.model.d_model)
    assert len(moe.experts) == tiny_cfg.model.n_experts


def test_route_weights_sum_to_one(gen):
    logits = torch.randn(11, 6, generator=gen, dtype=torch.float64)
    w, idx = route(logits, 2)
    assert w.shape == (11, 2) and idx.shape == (11, 2)
    assert torch.allclose(w.sum(-1), torch.ones(11, dtype=torch.float64), atol=1e-12)


def test_route_picks_top_experts():
    logits = torch.tensor([[0.1, 3.0, 2.0, -1.0], [5.0, 0.0, 1.0, 4.0], [-2.0, -1.0, -3.0, 0.5]], dtype=torch.float64)
    w, idx = route(logits, 2)
    assert [set(r) for r in idx.tolist()] == [{1, 2}, {0, 3}, {3, 1}]
    # веса: softmax только по выбранным логитам, в том же порядке, что и indices
    expected = torch.softmax(logits.gather(1, idx), dim=-1)
    assert torch.allclose(w, expected, atol=1e-12)
    assert torch.allclose(w[0].sum(), torch.tensor(1.0, dtype=torch.float64))


def test_aux_loss_uniform_routing_equals_alpha():
    logits = cyclic_logits(16, 8)
    _, idx = route(logits, 2)
    # sanity: нагрузка действительно равномерна
    assert torch.bincount(idx.reshape(-1), minlength=8).tolist() == [4] * 8
    aux = switch_aux_loss(logits, idx, 0.01)
    # P_e тоже равномерна (каждый эксперт один раз «первый», один раз «второй»), поэтому L = alpha
    assert abs(aux.item() - 0.01) < 1e-12


def test_aux_loss_penalizes_collapse():
    logits = torch.zeros(16, 4, dtype=torch.float64)
    logits[:, 0], logits[:, 1] = 3.0, 2.0  # все токены -> эксперты 0 и 1
    _, idx = route(logits, 2)
    aux = switch_aux_loss(logits, idx, 0.01)
    p = torch.softmax(logits, -1).mean(0)
    f = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=torch.float64)
    assert abs(aux.item() - 0.01 * 4 * (f * p).sum().item()) < 1e-12
    assert aux.item() > 0.01 * 1.2


def test_aux_loss_gradient_matches_analytic(gen):
    N, E, k, alpha = 12, 5, 2, 0.01
    logits = torch.randn(N, E, generator=gen, dtype=torch.float64).requires_grad_()
    _, idx = route(logits.detach(), k)
    switch_aux_loss(logits, idx, alpha).backward()
    p = torch.softmax(logits.detach(), -1)
    f = torch.bincount(idx.reshape(-1), minlength=E).double() / (k * N)
    # L = alpha*E*sum_e f_e P_e, f без градиента: dL/dl[t,j] = alpha*E/N * p[t,j] * (f_j - sum_e f_e p[t,e])
    expected = alpha * E / N * p * (f[None, :] - (p * f[None, :]).sum(-1, keepdim=True))
    assert torch.allclose(logits.grad, expected, atol=1e-12)


def test_last_load(tiny_cfg, gen):
    moe = build_moe(tiny_cfg)
    assert moe.last_load is None
    x = torch.randn(3, 7, tiny_cfg.model.d_model, generator=gen, dtype=torch.float64)
    moe(x)
    load = moe.last_load
    assert load.shape == (tiny_cfg.model.n_experts,)
    assert abs(load.sum().item() - 1.0) < 1e-6
    assert not load.requires_grad


def test_forward_matches_dense_reference(tiny_cfg, gen):
    moe = build_moe(tiny_cfg)
    x = torch.randn(2, 5, tiny_cfg.model.d_model, generator=gen, dtype=torch.float64)
    y, _ = moe(x)
    flat = x.reshape(-1, x.shape[-1])
    w, idx = route(moe.router(flat), tiny_cfg.model.top_k)
    ref = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(tiny_cfg.model.top_k):
            ref[t] += w[t, j] * moe.experts[int(idx[t, j])](flat[t])
    assert torch.allclose(y, ref.reshape(x.shape), atol=ATOL)


def test_gradients_reach_router_and_experts(tiny_cfg, gen):
    moe = build_moe(tiny_cfg)
    x = torch.randn(2, 5, tiny_cfg.model.d_model, generator=gen, dtype=torch.float64)
    y, aux = moe(x)
    (y.sum() + aux).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    assert sum(1 for e in moe.experts if e.w_up.weight.grad is not None) >= 1


def test_golden(tiny_cfg, golden):
    g = golden("moe")
    moe = build_moe(tiny_cfg)
    moe.load_state_dict(g["state_dict"])
    y, aux = moe(g["x"])
    assert torch.allclose(y, g["y"], atol=ATOL)
    assert torch.allclose(aux, g["aux"], atol=ATOL)
    assert torch.allclose(moe.last_load.double(), g["load"], atol=1e-7)
    w, idx = route(g["route_logits"], 2)
    assert torch.equal(idx, g["route_indices"]) and torch.allclose(w, g["route_weights"], atol=ATOL)
    assert torch.allclose(switch_aux_loss(g["route_logits"], idx, 0.01), g["route_aux"], atol=ATOL)
