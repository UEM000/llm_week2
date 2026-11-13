import torch
from torch import nn

from kestrel.model.yarn import yarn_inv_freq, yarn_mscale


def rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    """theta_i = base^(-2i/Dh), i = 0..Dh/2-1. Форма [Dh/2], float32."""
    i = torch.arange(0, head_dim // 2, dtype=torch.float32)
    return base ** (-2.0 * i / head_dim)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """RoPE в конвенции rotate_half. x: [B, h, T, Dh], positions: [T] (long, абсолютные).

    out1 = x1*cos(p*theta) - x2*sin(p*theta), out2 = x2*cos(p*theta) + x1*sin(p*theta),
    где x1, x2 -- первая и вторая половины последней оси.
    """
    angles = positions.to(torch.float32)[:, None] * inv_freq.to(torch.float32)[None, :]  # [T, Dh/2]
    cos = angles.cos().to(x.dtype)
    sin = angles.sin().to(x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RotaryEmbedding(nn.Module):
    """Общий для всех слоёв RoPE с поддержкой YaRN. Масштаб меняется через set_scale()."""

    def __init__(self, head_dim: int, base: float, l_train: int, alpha: float, beta: float):
        super().__init__()
        self.l_train, self.alpha, self.beta = l_train, alpha, beta
        # частоты считаем в float64 на CPU: результат не зависит от устройства и от порядка операций
        with torch.device("cpu"):
            self._base_inv_freq = rope_inv_freq(head_dim, base).double()
            self.register_buffer("inv_freq", rope_inv_freq(head_dim, base), persistent=False)
        self.scale = 1.0  # без YaRN: частоты исходные, mscale = 1 (yarn.py понадобится только при set_scale)
        self.mscale = 1.0

    def set_scale(self, scale: float) -> None:
        """Пересчитывает частоты (YaRN) и множитель внимания для нового scale = L_new / L_train."""
        f = yarn_inv_freq(self._base_inv_freq, scale, self.l_train, self.alpha, self.beta)
        self.inv_freq = f.to(device=self.inv_freq.device, dtype=torch.float32)
        self.scale = scale
        self.mscale = yarn_mscale(scale)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return apply_rope(x, positions, self.inv_freq)
