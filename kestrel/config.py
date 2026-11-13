"""Конфигурация задания: читается из configs/kestrel.yaml, хеш файла проверяется."""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "kestrel.yaml"


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_experts: int
    top_k: int
    d_ff: int
    rope_base: float
    norm_eps: float
    init_std: float


@dataclass
class YarnConfig:
    l_train: int
    alpha: float
    beta: float


@dataclass
class MoEConfig:
    aux_alpha: float


@dataclass
class OptimConfig:
    betas: tuple
    eps: float
    weight_decay: float
    grad_clip: float
    muon_momentum: float
    muon_nesterov: bool
    muon_ns_steps: int


@dataclass
class StageConfig:
    seq_len: int
    batch_size: int
    steps: int
    lr: float
    warmup_steps: int
    decay_frac: float
    final_lr_frac: float
    yarn_scale: float
    micro_batch_size: int | None = None  # None -> весь батч за один проход
    grad_accum: int = 1                  # batch_size = micro_batch_size * grad_accum
    reset_optimizer_state: bool = True
    ckpt_every: int = 250


@dataclass
class SeedsConfig:
    tests: int
    init: int
    data_order: int
    passkey_eval: int


@dataclass
class DataConfig:
    hf_repo: str
    revision: str


@dataclass
class CheckpointsConfig:
    hf_repo: str
    revision: str
    files: dict
    val_loss_tolerance: float
    reference_val_loss: dict


@dataclass
class SpeedCheckConfig:
    finetune_step_warn_sec: float


@dataclass
class PrecisionConfig:
    autocast_dtype: str
    optimizer_dtype: str


@dataclass
class Config:
    model: ModelConfig
    yarn: YarnConfig
    moe: MoEConfig
    optim: OptimConfig
    pretrain: StageConfig
    finetune: StageConfig
    seeds: SeedsConfig
    data: DataConfig
    checkpoints: CheckpointsConfig
    speed_check: SpeedCheckConfig
    precision: PrecisionConfig


def _build(cls, raw: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные ключи конфига {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in raw:
            value = raw[f.name]
            kwargs[f.name] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


def _resolve(path) -> Path:
    return Path(path) if path is not None else DEFAULT_CONFIG_PATH


def load_config(path=None) -> Config:
    """Читает YAML и возвращает Config (вложенные секции доступны как атрибуты)."""
    path = _resolve(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(**{name: _build(cls, raw[name]) for name, cls in _SECTION_TYPES.items()})
    for stage_name in ("pretrain", "finetune"):
        st = getattr(cfg, stage_name)
        if st.micro_batch_size is not None and st.micro_batch_size * st.grad_accum != st.batch_size:
            raise ValueError(f"{stage_name}: batch_size={st.batch_size} != micro_batch_size*grad_accum="
                             f"{st.micro_batch_size * st.grad_accum}")
    return cfg


_SECTION_TYPES = {
    "model": ModelConfig,
    "yarn": YarnConfig,
    "moe": MoEConfig,
    "optim": OptimConfig,
    "pretrain": StageConfig,
    "finetune": StageConfig,
    "seeds": SeedsConfig,
    "data": DataConfig,
    "checkpoints": CheckpointsConfig,
    "speed_check": SpeedCheckConfig,
    "precision": PrecisionConfig,
}


def compute_config_hash(path=None) -> str:
    """SHA-256 файла конфига; переводы строк CRLF/LF не влияют на результат."""
    data = _resolve(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def verify_config_hash(path=None) -> None:
    """Сверяет хеш конфига с configs/kestrel.yaml.sha256. Бросает RuntimeError при расхождении."""
    path = _resolve(path)
    hash_file = path.with_name(path.name + ".sha256")
    if not hash_file.exists():
        raise RuntimeError(f"Не найден файл с хешем конфига: {hash_file}")
    expected = hash_file.read_text(encoding="utf-8").split()[0].strip().lower()
    actual = compute_config_hash(path)
    if actual != expected:
        raise RuntimeError(
            f"Конфиг {path.name} изменён: sha256={actual}, ожидалось {expected}. "
            "Верните исходный configs/kestrel.yaml (менять его в задании нельзя)."
        )


def tiny_config() -> Config:
    """Маленький конфиг для тестов; всё, кроме размеров модели, как в основном."""
    cfg = load_config()
    cfg.model = dataclasses.replace(
        cfg.model,
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        n_experts=4,
        top_k=2,
        d_ff=64,
    )
    return cfg


def smoke_config() -> Config:
    """Крошечный конфиг для сквозного прогона на CPU (режим SMOKE в ноутбуке)."""
    cfg = tiny_config()
    cfg.yarn = dataclasses.replace(cfg.yarn, l_train=32)
    cfg.pretrain = dataclasses.replace(
        cfg.pretrain, seq_len=32, batch_size=8, micro_batch_size=None, grad_accum=1, steps=30, lr=3.0e-3,
        warmup_steps=3, ckpt_every=250,
    )
    cfg.finetune = dataclasses.replace(
        cfg.finetune, seq_len=128, batch_size=4, micro_batch_size=2, grad_accum=2, steps=20, lr=2.0e-3,
        warmup_steps=2, ckpt_every=10,
    )
    return cfg
