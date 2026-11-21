import torch
from torch import nn

from kestrel.model.expert import SwiGLUExpert


def route(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Выбор top-k экспертов для каждого токена.

    logits: [N, E] -- логиты роутера. Возвращает (weights [N, k], indices [N, k]); weights[n, j]
    -- вес эксперта indices[n, j], веса токена в сумме дают 1. Формулы: TECH_REPORT.md, раздел 2.4;
    приложение A.3.
    """
    vals, idx = logits.topk(k=top_k, dim=-1)
    weights = torch.softmax(vals, dim=-1)
    return weights, idx

def frac_of_tokens(indices: torch.Tensor, E: int):
    N, k = indices.shape
    return (
            indices
            .flatten()
            .bincount(minlength=E)
        ) / (k * N)

def switch_aux_loss(logits: torch.Tensor, indices: torch.Tensor, alpha: float) -> torch.Tensor:
    """Балансировочный лосс Switch Transformer -- скаляр.

    logits: [N, E] -- логиты роутера; indices: [N, k] -- выбранные эксперты (из route).
    Формулы: TECH_REPORT.md, раздел 2.4; приложение A.3.
    """
    _, E = logits.shape
    probs = torch.softmax(logits, dim=-1)
    P = probs.mean(dim=0)
    f = frac_of_tokens(indices, E)

    return (P @ f) * alpha * E


class MoE(nn.Module):
    """Смесь из E SwiGLU-экспертов с top-k роутингом (без capacity, токены не выбрасываются).

    Готовые атрибуты (имена не менять): router (Linear(D, E) без bias), experts (ModuleList из
    SwiGLUExpert), last_load (None до первого forward). Описание: TECH_REPORT.md, раздел 2.4;
    приложение A.3.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int, aux_alpha: float):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_alpha = aux_alpha
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(n_experts)])
        self.last_load = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, D] -> (y [B, T, D], aux_loss -- скаляр).

        Побочный эффект: self.last_load = [E] -- доля токен-слотов у каждого эксперта на этом
        forward (detached, сумма равна 1). Логиты роутера считаются минимум в fp32. Алгоритм:
        TECH_REPORT.md, раздел 2.4 и приложение A.3.
        """
        B, T, D = x.shape
        x = x.view(B * T, D)
        logits: torch.Tensor = self.router(x)
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))

        weights, idx = route(logits, self.top_k)
        y = torch.zeros_like(x)

        for e in range(self.n_experts):
            rows, cols = (idx == e).nonzero(as_tuple=True)
            if rows is None or cols is None:
                continue
            out: torch.Tensor = self.experts[e](x[rows])
            y.index_add_(dim=0, index=rows, source=out * weights[rows, cols, None].to(out.dtype))

        self.last_load = frac_of_tokens(idx, logits.shape[-1]).detach()
        loss = switch_aux_loss(logits, idx, self.aux_alpha)
        y = y.view(B, T, D)

        return y, loss
