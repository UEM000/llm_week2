import math

import torch

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(m: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Приближённая ортогонализация матрицы итерациями Ньютона-Шульца (в fp32)."""
    a, b, c = NS_COEFFS
    x = m.float()
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * (A @ A)) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon для 2D-параметров: момент -> ортогонализация (Newton-Schulz) -> шаг.

        B ← μB + G;  M = G + μB (nesterov) или B
        O = NS(M)
        W ← W − lr·(0.2·sqrt(max(rows, cols))·O + wd·W)
    """

    def __init__(self, params, lr, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError("Muon работает только с 2D-параметрами")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                m = p.grad.add(buf, alpha=mu) if group["nesterov"] else buf
                o = newton_schulz(m, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * math.sqrt(max(p.shape))
                p.add_(scale * o + wd * p, alpha=-lr)
        return loss
