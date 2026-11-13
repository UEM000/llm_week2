import pytest
import torch

from kestrel.model.attention import GQAttention, expand_kv
from kestrel.model.kv_cache import KVCache
from kestrel.model.rope import RotaryEmbedding

ATOL = 1e-10


def build_attn(cfg, n_kv_heads=None, randomize_norms=True):
    m = cfg.model
    rotary = RotaryEmbedding(m.head_dim, m.rope_base, cfg.yarn.l_train, cfg.yarn.alpha, cfg.yarn.beta)
    attn = GQAttention(m.d_model, m.n_heads, n_kv_heads or m.n_kv_heads, m.head_dim, m.norm_eps, rotary)
    if randomize_norms:
        with torch.no_grad():
            attn.q_norm.weight.add_(torch.rand(m.head_dim) * 0.5)
            attn.k_norm.weight.add_(torch.rand(m.head_dim) * 0.5)
    return attn


def rand_x(cfg, gen, batch=2, seq=9):
    return torch.randn(batch, seq, cfg.model.d_model, generator=gen, dtype=torch.float64)


def test_attribute_names(tiny_cfg):
    attn = build_attn(tiny_cfg)
    m = tiny_cfg.model
    assert attn.q_proj.weight.shape == (m.n_heads * m.head_dim, m.d_model)
    assert attn.k_proj.weight.shape == (m.n_kv_heads * m.head_dim, m.d_model)
    assert attn.v_proj.weight.shape == (m.n_kv_heads * m.head_dim, m.d_model)
    assert attn.o_proj.weight.shape == (m.d_model, m.n_heads * m.head_dim)
    assert attn.q_norm.weight.shape == (m.head_dim,) and attn.k_norm.weight.shape == (m.head_dim,)
    assert attn.q_proj.bias is None and attn.o_proj.bias is None


def test_output_shape(tiny_cfg, gen):
    attn = build_attn(tiny_cfg)
    x = rand_x(tiny_cfg, gen)
    assert attn(x, 0).shape == x.shape


def test_causality(tiny_cfg, gen):
    attn = build_attn(tiny_cfg)
    x = rand_x(tiny_cfg, gen)
    x2 = x.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:])
    out1, out2 = attn(x, 0), attn(x2, 0)
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-12)
    assert not torch.allclose(out1[:, 5:], out2[:, 5:], atol=1e-6)


def test_expand_kv_mapping():
    B, Hkv, T, Dh = 2, 2, 3, 4
    x = torch.arange(B * Hkv * T * Dh, dtype=torch.float64).reshape(B, Hkv, T, Dh)
    out = expand_kv(x, 6)
    assert out.shape == (B, 6, T, Dh)
    for h in range(6):
        assert torch.equal(out[:, h], x[:, h // 3])
    assert torch.equal(expand_kv(x, 2), x)
    assert torch.equal(expand_kv(x[:, :1], 4), x[:, :1].expand(B, 4, T, Dh))


@pytest.mark.parametrize("n_kv", [1, 2])
def test_gqa_equals_mha_with_tied_weights(tiny_cfg, gen, n_kv):
    m = tiny_cfg.model
    gqa = build_attn(tiny_cfg, n_kv_heads=n_kv)
    mha = build_attn(tiny_cfg, n_kv_heads=m.n_heads)
    group = m.n_heads // n_kv
    sd = {k: v.clone() for k, v in gqa.state_dict().items()}
    for name in ("k_proj.weight", "v_proj.weight"):
        w = sd[name].view(n_kv, m.head_dim, m.d_model).repeat_interleave(group, dim=0)
        sd[name] = w.reshape(m.n_heads * m.head_dim, m.d_model)
    mha.load_state_dict(sd)
    x = rand_x(tiny_cfg, gen)
    for scale in (1.0, 4.0):
        gqa.rotary.set_scale(scale)
        mha.rotary.set_scale(scale)
        assert torch.allclose(gqa(x, 0), mha(x, 0), atol=ATOL)


def test_incremental_decoding_equals_full_forward(tiny_cfg, gen):
    attn = build_attn(tiny_cfg)
    x = rand_x(tiny_cfg, gen, seq=10)
    full = attn(x, 0)
    cache = KVCache(1)
    outs = [attn(x[:, :4], 0, cache)]
    for t in range(4, 10):
        outs.append(attn(x[:, t : t + 1], 0, cache))
    assert cache.seq_len(0) == 10
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=ATOL)


def test_chunked_prefill_equals_full_prefill(tiny_cfg, gen):
    attn = build_attn(tiny_cfg)
    x = rand_x(tiny_cfg, gen, seq=9)
    full = attn(x, 0, KVCache(1))
    cache = KVCache(1)
    outs, start = [], 0
    for size in (3, 2, 4):
        outs.append(attn(x[:, start : start + size], 0, cache))
        start += size
    assert torch.allclose(torch.cat(outs, dim=1), full, atol=ATOL)
    assert torch.allclose(full, attn(x, 0), atol=ATOL)


def test_cache_stores_post_rope_keys(tiny_cfg, gen):
    """Состояние кэша не должно зависеть от разбиения на куски (иначе в кэше лежат k до RoPE)."""
    attn = build_attn(tiny_cfg)
    x = rand_x(tiny_cfg, gen, seq=6)
    c1, c2 = KVCache(1), KVCache(1)
    attn(x, 0, c1)
    attn(x[:, :2], 0, c2)
    attn(x[:, 2:], 0, c2)
    assert torch.allclose(c1.k[0], c2.k[0], atol=ATOL) and torch.allclose(c1.v[0], c2.v[0], atol=ATOL)


def test_golden_forward_and_yarn(tiny_cfg, golden):
    g = golden("attention")
    attn = build_attn(tiny_cfg, randomize_norms=False)
    attn.load_state_dict(g["state_dict"])
    assert torch.allclose(attn(g["x"], 0), g["out_scale1"], atol=ATOL)
    attn.rotary.set_scale(4.0)
    out4 = attn(g["x"], 0)
    assert torch.allclose(out4, g["out_scale4"], atol=ATOL)
    assert not torch.allclose(out4, g["out_scale1"], atol=1e-3)
