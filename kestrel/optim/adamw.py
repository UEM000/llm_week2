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
        raise NotImplementedError("TODO: AdamW.step")
