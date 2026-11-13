import torch
from torch import nn


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2, -1) + eps) * weight; считается в fp32, результат в dtype входа."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.to(torch.promote_types(dtype, torch.float32))
        y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight).to(dtype)
