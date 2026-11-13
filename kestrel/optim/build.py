from typing import Literal

from kestrel.model.model import Kestrel
from kestrel.optim.adamw import AdamW
from kestrel.optim.muon import Muon
from kestrel.optim.param_groups import split_params


class CompositeOptimizer:
    """Тонкая обёртка над несколькими оптимизаторами: общий step / zero_grad / set_lr / state_dict."""

    def __init__(self, optimizers: list):
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list:
        """Все param groups подряд (нужно GradScaler.unscale_ и клиппингу)."""
        return [g for opt in self.optimizers for g in opt.param_groups]

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def set_lr(self, lr: float) -> None:
        for group in self.param_groups:
            group["lr"] = lr

    def state_dict(self) -> dict:
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state: dict) -> None:
        for opt, s in zip(self.optimizers, state["optimizers"], strict=True):
            opt.load_state_dict(s)


def build_optimizer(model: Kestrel, kind: Literal["adamw", "muon"], stage_cfg, optim_cfg) -> CompositeOptimizer:
    """adamw: три группы на AdamW; muon: 'matrix' на Muon, остальные две на AdamW."""
    groups = split_params(model)
    wd = optim_cfg.weight_decay
    adam_kwargs = dict(lr=stage_cfg.lr, betas=tuple(optim_cfg.betas), eps=optim_cfg.eps)
    decay_groups = [
        {"params": groups["vector_decay"], "weight_decay": wd},
        {"params": groups["no_decay"], "weight_decay": 0.0},
    ]
    if kind == "adamw":
        return CompositeOptimizer(
            [AdamW([{"params": groups["matrix"], "weight_decay": wd}] + decay_groups, **adam_kwargs)]
        )
    if kind == "muon":
        muon = Muon(
            groups["matrix"],
            lr=stage_cfg.lr,
            momentum=optim_cfg.muon_momentum,
            nesterov=optim_cfg.muon_nesterov,
            ns_steps=optim_cfg.muon_ns_steps,
            weight_decay=wd,
        )
        return CompositeOptimizer([muon, AdamW(decay_groups, **adam_kwargs)])
    raise ValueError(f"Неизвестный оптимизатор: {kind!r} (ожидается 'adamw' или 'muon')")
