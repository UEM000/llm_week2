import math

import torch
import torch.nn.functional as F
from torch import nn

from kestrel.model.kv_cache import KVCache
from kestrel.model.norm import RMSNorm
from kestrel.model.rope import RotaryEmbedding


def expand_kv(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Размножает KV-головы до числа голов запросов: [B, Hkv, T, Dh] -> [B, H, T, Dh].

    Какая KV-голова достаётся голове запросов h, описано в TECH_REPORT.md, раздел 2.2; приложение A.2.
    Если n_heads не делится на Hkv, бросает ValueError.
    """
    _, Hkv, _, _ = x.shape
    if Hkv == n_heads:
        return x
    if n_heads % Hkv != 0:
        raise ValueError("Incorrect KV-heads ammoun.")
    repeats = n_heads // Hkv
    return torch.repeat_interleave(x, repeats, dim=1)


class GQAttention(nn.Module):
    """Grouped-Query Attention с QK-Norm и RoPE/YaRN (MQA: n_kv_heads = 1, MHA: n_kv_heads = n_heads).

    Готовые атрибуты (имена не менять -- по ним загружаются веса в тестах):
    q_proj, k_proj, v_proj, o_proj (Linear без bias), q_norm, k_norm (RMSNorm по оси головы),
    rotary -- общий RotaryEmbedding модели (ссылка, не копия).

    Описание: TECH_REPORT.md, раздел 2.2; пошаговый алгоритм -- приложение A.2.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        rotary: RotaryEmbedding,
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads должно делиться на n_kv_heads"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.q_norm = RMSNorm(head_dim, norm_eps)
        self.k_norm = RMSNorm(head_dim, norm_eps)
        self.rotary = rotary

    def forward(self, x: torch.Tensor, layer_idx: int, cache: KVCache | None = None) -> torch.Tensor:
        """x: [B, T, D] -> [B, T, D].

        cache -- KVCache или None. С кэшем новые k, v дописываются в него, а внимание смотрит на весь
        накопленный префикс; должны работать и prefill (T > 1, в том числе при непустом кэше), и
        декодирование (T = 1).

        Внимание считается ТОЛЬКО через F.scaled_dot_product_attention(q, k, v, attn_mask=..., is_causal=...,
        scale=self.rotary.mscale ** 2 / math.sqrt(head_dim)):
        - offset == 0 -> is_causal=True, attn_mask=None;
        - offset > 0  -> булева маска [T, T_k], True = «можно смотреть», is_causal=False;
        - scale передаётся обязательно (иначе теряется m^2).
        Алгоритм: TECH_REPORT.md, раздел 2.2 и приложение A.2.
        """
        offset = 0 if cache is None else cache.seq_len(layer_idx)
        B, T, D = x.shape

        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(x)
        v: torch.Tensor = self.v_proj(x)

        # [B, T, D] -> [B, T, H, Dh].
        q = q.view(B, T, self.n_heads, self.head_dim)
        # [B, T, H_kv, D_h]
        k = k.view(B, T, self.n_kv_heads, self.head_dim)
        v = v.view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        T_k = offset + T
        positions = torch.arange(offset, T_k, device=x.device)
        q = self.rotary(q, positions)
        k = self.rotary(k, positions)

        if cache is not None: k, v = cache.update(layer_idx, k, v)

        k = expand_kv(k, self.n_heads)
        v = expand_kv(v, self.n_heads)

        if offset == 0:
            is_causal = True
            attn_mask = None
        else:
            is_causal = False
            attn_mask = torch.arange(T_k, device=x.device)[None, :] <= positions[:, None]
        logits = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask, is_causal=is_causal,
            scale=self.rotary.mscale ** 2 / math.sqrt(self.head_dim)
        )
        logits = logits.transpose(1, 2).contiguous()
        logits = logits.view(B, T, D)
        return self.o_proj(logits)