import math

import torch

from kestrel.model.rope import rope_inv_freq
from kestrel.model.yarn import yarn_inv_freq, yarn_mscale

L_TRAIN, ALPHA, BETA = 512, 1.0, 32.0


def base_freqs(dtype=torch.float64, head_dim=32):
    return rope_inv_freq(head_dim, 10000.0).to(dtype)


def ratios(scale, dtype=torch.float64):
    inv = base_freqs(dtype)
    return yarn_inv_freq(inv, scale, L_TRAIN, ALPHA, BETA) / inv


def test_scale_one_is_exact_identity():
    for dtype in (torch.float32, torch.float64):
        inv = base_freqs(dtype)
        out = yarn_inv_freq(inv, 1.0, L_TRAIN, ALPHA, BETA)
        assert torch.equal(out, inv)
    assert yarn_mscale(1.0) == 1.0


def test_high_frequencies_unchanged():
    r = ratios(4.0)
    assert torch.allclose(r[:2], torch.ones(2), atol=1e-12)  # r_i > beta


def test_low_frequencies_divided_by_scale():
    for scale in (2.0, 4.0, 8.0):
        r = ratios(scale)
        assert torch.allclose(r[8:], torch.full((8,), 1.0 / scale), atol=1e-12)  # r_i < alpha


def test_ramp_is_monotonic_and_between():
    r = ratios(4.0)
    assert torch.all(r[1:] <= r[:-1] + 1e-15)
    assert torch.all(r[2:8] < 1.0) and torch.all(r[2:8] > 0.25)


def test_ramp_values_match_formula():
    inv = base_freqs()
    out = yarn_inv_freq(inv, 4.0, L_TRAIN, ALPHA, BETA)
    for i, theta in enumerate(inv.tolist()):
        r_i = L_TRAIN * theta / (2 * math.pi)
        gamma = min(max((r_i - ALPHA) / (BETA - ALPHA), 0.0), 1.0)
        expected = (1 - gamma) * theta / 4.0 + gamma * theta
        assert abs(out[i].item() - expected) < 1e-12


def test_mscale():
    assert abs(yarn_mscale(4.0) - 1.138629436) < 1e-6
    assert isinstance(yarn_mscale(4.0), float)


def test_dtype_and_shape_preserved():
    for dtype in (torch.float32, torch.float64):
        inv = base_freqs(dtype)
        out = yarn_inv_freq(inv, 4.0, L_TRAIN, ALPHA, BETA)
        assert out.dtype == dtype and out.shape == inv.shape


def test_input_not_modified():
    inv = base_freqs()
    before = inv.clone()
    yarn_inv_freq(inv, 4.0, L_TRAIN, ALPHA, BETA)
    assert torch.equal(inv, before)


def test_golden(golden):
    for case in golden("yarn")["cases"]:
        out = yarn_inv_freq(case["inv_freq"], case["scale"], case["l_train"], case["alpha"], case["beta"])
        assert torch.allclose(out, case["expected"], atol=1e-10, rtol=0)
