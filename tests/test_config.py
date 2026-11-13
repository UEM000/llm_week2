import pytest

from kestrel.config import DEFAULT_CONFIG_PATH, compute_config_hash, tiny_config, verify_config_hash


def test_config_hash_matches():
    verify_config_hash()


def test_config_hash_mismatch_raises(tmp_path):
    src = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    p = tmp_path / "kestrel.yaml"
    p.write_text(src.replace("n_layers: 8", "n_layers: 9"), encoding="utf-8")
    (tmp_path / "kestrel.yaml.sha256").write_text(compute_config_hash() + "  kestrel.yaml\n")
    with pytest.raises(RuntimeError):
        verify_config_hash(p)


def test_key_values(full_cfg):
    m = full_cfg.model
    assert (m.vocab_size, m.d_model, m.n_layers) == (16384, 256, 8)
    assert (m.n_heads, m.n_kv_heads, m.head_dim) == (8, 2, 32)
    assert (m.n_experts, m.top_k, m.d_ff) == (8, 2, 256)
    assert (full_cfg.yarn.l_train, full_cfg.yarn.alpha, full_cfg.yarn.beta) == (512, 1.0, 32.0)
    assert full_cfg.moe.aux_alpha == 0.01
    assert tuple(full_cfg.optim.betas) == (0.9, 0.95)
    assert full_cfg.optim.weight_decay == 0.1
    assert full_cfg.pretrain.seq_len == 512 and full_cfg.finetune.seq_len == 2048
    assert full_cfg.finetune.yarn_scale == 4.0
    assert (full_cfg.seeds.tests, full_cfg.seeds.init, full_cfg.seeds.data_order, full_cfg.seeds.passkey_eval) == (
        0, 1337, 2024, 7)


def test_tiny_config(tiny_cfg):
    m = tiny_cfg.model
    assert (m.d_model, m.n_layers, m.n_heads, m.n_kv_heads, m.head_dim) == (64, 2, 4, 2, 16)
    assert (m.n_experts, m.top_k, m.d_ff, m.vocab_size) == (4, 2, 64, 128)
    assert m.rope_base == 10000.0 and tiny_cfg.moe.aux_alpha == 0.01
