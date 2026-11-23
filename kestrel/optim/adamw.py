import math

import torch


class AdamW(torch.optim.Optimizer):
    """AdamW с декаплированным weight decay. Аргументы и значения по умолчанию как у torch.optim.AdamW.

    `__init__` готов: он проверяет аргументы и складывает lr, betas, eps, weight_decay в `defaults`
    (у каждой param group они свои). Вам нужно написать `step`.

    Состояние на параметр создаётся лениво при первом шаге: 'step', 'exp_avg', 'exp_avg_sq'.
    Контракт: совпадает с torch.optim.AdamW(foreach=False) с точностью до 1e-6 (fp32).

    Формулы и порядок действий: TECH_REPORT.md, раздел 4.1; приложение A.4.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0:
            raise ValueError(f"Некорректный lr: {lr}")
        if eps < 0.0:
            raise ValueError(f"Некорректный eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Некорректные betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Некорректный weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Один шаг оптимизации. closure -- функция, заново считающая loss и градиенты (необязательна);
        если она передана, шаг возвращает её значение, иначе None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1

                step = state["step"]
                m: torch.Tensor = state["exp_avg"]
                v: torch.Tensor = state["exp_avg_sq"]

                # new_m = beta1 * m + (1 - beta1) * g = beta1 * m + (1 - beta1) * g +- m = 
                # = m - (1 - beta) * m + (1 - beta1) * g = m + (1 - beta1) * (g - m) =>
                #  => m += (1 - beta1) * (g - m)
                m.lerp_(g, 1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
        
                denom = (v.sqrt() / bc2 ** 0.5).add_(eps)
                p.addcdiv_(m, denom, value=-(lr / bc1))

        return loss