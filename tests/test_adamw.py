import copy

import pytest
import torch
from torch import nn

from kestrel.optim.adamw import AdamW

BETAS, EPS = (0.9, 0.95), 1e-4  # большой eps, чтобы отличать eps под корнем


def make_mlp(gen):
    net = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4)).float()
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(*p.shape, generator=gen, dtype=torch.float32) * 0.3)
    return net


def make_data(gen):
    x = torch.randn(32, 8, generator=gen, dtype=torch.float32)
    y = torch.randn(32, 4, generator=gen, dtype=torch.float32)
    return x, y


def groups_of(net):
    return [
        {"params": list(net[0].parameters()), "lr": 1e-3, "weight_decay": 0.1},
        {"params": list(net[2].parameters()), "lr": 5e-3, "weight_decay": 0.0},
    ]


def train_step(net, opt, x, y):
    opt.zero_grad()
    loss = ((net(x) - y) ** 2).mean()
    loss.backward()
    opt.step()


def assert_close(net_a, net_b, atol):
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.allclose(pa, pb, atol=atol, rtol=0), (pa - pb).abs().max()


def test_matches_torch_adamw_two_groups_and_lr_change(gen):
    net_a = make_mlp(gen)
    x, y = make_data(gen)
    net_b = copy.deepcopy(net_a)
    ours = AdamW(groups_of(net_a), betas=BETAS, eps=EPS)
    ref = torch.optim.AdamW(groups_of(net_b), betas=BETAS, eps=EPS, foreach=False)
    for step in range(100):
        if step == 50:
            for opt in (ours, ref):
                for g in opt.param_groups:
                    g["lr"] *= 0.1
        train_step(net_a, ours, x, y)
        train_step(net_b, ref, x, y)
    assert_close(net_a, net_b, 1e-6)


def test_default_hyperparameters_match_torch():
    p = nn.Parameter(torch.zeros(2))
    g = AdamW([p]).param_groups[0]
    r = torch.optim.AdamW([p]).param_groups[0]
    for key in ("lr", "betas", "eps", "weight_decay"):
        assert g[key] == r[key], key


def test_invalid_arguments_rejected():
    p = nn.Parameter(torch.zeros(2))
    for kwargs in ({"lr": -1.0}, {"eps": -1e-8}, {"betas": (1.0, 0.9)}, {"weight_decay": -0.1}):
        with pytest.raises(ValueError):
            AdamW([p], **kwargs)


def test_parameter_without_grad_is_untouched(gen):
    net = make_mlp(gen)
    extra = nn.Parameter(torch.ones(3))
    opt = AdamW([{"params": list(net.parameters()) + [extra]}], lr=1e-2)
    x, y = make_data(gen)
    for _ in range(3):
        train_step(net, opt, x, y)
    assert torch.equal(extra.detach(), torch.ones(3))
    assert extra not in opt.state or len(opt.state[extra]) == 0


def test_per_parameter_step_counter(gen):
    """Параметр, у которого градиент появился позже, должен иметь собственный счётчик шагов (bias correction)."""
    a, b = nn.Parameter(torch.randn(5, generator=gen)), nn.Parameter(torch.randn(5, generator=gen))
    a2, b2 = nn.Parameter(a.detach().clone()), nn.Parameter(b.detach().clone())
    ours = AdamW([a, b], lr=1e-2, betas=BETAS, weight_decay=0.1)
    ref = torch.optim.AdamW([a2, b2], lr=1e-2, betas=BETAS, weight_decay=0.1, foreach=False)
    target = torch.linspace(-1, 1, 5)
    for t in range(40):
        for opt, (p, q) in ((ours, (a, b)), (ref, (a2, b2))):
            opt.zero_grad()
            loss = ((p - target) ** 2).sum()
            if t >= 10:
                loss = loss + ((q + target) ** 2).sum()
            loss.backward()
            opt.step()
    assert torch.allclose(a, a2, atol=1e-6, rtol=0) and torch.allclose(b, b2, atol=1e-6, rtol=0)


def test_state_dict_roundtrip_continues_trajectory(gen):
    net_a = make_mlp(gen)
    x, y = make_data(gen)
    opt_a = AdamW(groups_of(net_a), betas=BETAS, eps=EPS)
    for _ in range(30):
        train_step(net_a, opt_a, x, y)
    net_b = copy.deepcopy(net_a)
    opt_b = AdamW(groups_of(net_b), betas=BETAS, eps=EPS)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))
    for _ in range(20):
        train_step(net_a, opt_a, x, y)
        train_step(net_b, opt_b, x, y)
    assert_close(net_a, net_b, 1e-7)


def test_state_has_expected_keys(gen):
    net = make_mlp(gen)
    opt = AdamW(net.parameters(), lr=1e-3)
    x, y = make_data(gen)
    train_step(net, opt, x, y)
    state = opt.state[next(net.parameters())]
    assert {"step", "exp_avg", "exp_avg_sq"} <= set(state.keys())
    assert float(state["step"]) == 1


def test_closure(gen):
    net_a = make_mlp(gen)
    net_b = copy.deepcopy(net_a)
    x, y = make_data(gen)
    ours = AdamW(net_a.parameters(), lr=1e-2)
    ref = torch.optim.AdamW(net_b.parameters(), lr=1e-2, foreach=False)

    def make_closure(net, opt):
        def closure():
            opt.zero_grad()
            loss = ((net(x) - y) ** 2).mean()
            loss.backward()
            return loss

        return closure

    for _ in range(5):
        l_ours = ours.step(make_closure(net_a, ours))
        l_ref = ref.step(make_closure(net_b, ref))
        assert l_ours is not None and abs(l_ours.item() - l_ref.item()) < 1e-6
    assert_close(net_a, net_b, 1e-6)
