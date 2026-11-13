from pathlib import Path

from kestrel.config import Config


def download_data(cfg: Config, local_dir, role: str = "student") -> Path:
    """Скачивает датасет с HuggingFace Hub (фиксированная ревизия из конфига) в local_dir.

    role="student": всё, кроме pretrain_train.bin (обучающий сплит претрейна студентам не нужен);
    role="teacher": всё.
    """
    from huggingface_hub import snapshot_download

    if role not in ("student", "teacher"):
        raise ValueError(f"role должен быть 'student' или 'teacher', получено {role!r}")
    if cfg.data.hf_repo.startswith("TBD") or cfg.data.revision == "TBD":
        raise RuntimeError(
            "В configs/kestrel.yaml не заполнены data.hf_repo / data.revision -- "
            "их выдаёт преподаватель."
        )
    path = snapshot_download(
        repo_id=cfg.data.hf_repo,
        repo_type="dataset",
        revision=cfg.data.revision,
        local_dir=str(local_dir),
        ignore_patterns=["pretrain_train.bin"] if role == "student" else None,
    )
    return Path(path)


def download_checkpoints(cfg: Config, local_dir) -> Path:
    """Скачивает чекпоинты претрейна (ckpt_A.pt, ckpt_M.pt, init.pt, логи, reference.json) в local_dir."""
    from huggingface_hub import snapshot_download

    if cfg.checkpoints.hf_repo.startswith("TBD") or cfg.checkpoints.revision == "TBD":
        raise RuntimeError("чекпоинты ещё не опубликованы: в configs/kestrel.yaml не заполнены "
                           "checkpoints.hf_repo / checkpoints.revision.")
    path = snapshot_download(
        repo_id=cfg.checkpoints.hf_repo,
        repo_type="model",
        revision=cfg.checkpoints.revision,
        local_dir=str(local_dir),
    )
    return Path(path)
