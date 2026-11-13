import hashlib
import platform
import subprocess
from pathlib import Path

import torch

from kestrel.config import compute_config_hash
from kestrel.data.loader import make_eval_loader
from kestrel.optim.param_groups import split_params

TODO_FILES = (
    "kestrel/model/yarn.py",
    "kestrel/model/attention.py",
    "kestrel/model/moe.py",
    "kestrel/optim/adamw.py",
    "kestrel/optim/param_groups.py",
)


def stable_rank(W: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2 -- «эффективное число направлений» матрицы."""
    s = torch.linalg.svdvals(W.detach().float().cpu())
    return float((s.pow(2).sum() / s[0].pow(2)).item())


def relative_drift(W: torch.Tensor, W0: torch.Tensor) -> float:
    """||W - W0||_F / ||W0||_F."""
    W, W0 = W.detach().float().cpu(), W0.detach().float().cpu()
    return float(((W - W0).norm() / W0.norm()).item())


def matrix_report(model, ref_state: dict | None = None) -> dict:
    """Метрики по всем 2D-матрицам группы 'matrix': {'per_matrix': {имя: {...}}, 'mean_stable_rank', 'mean_drift'}.

    ref_state -- state_dict модели, от которой считается дрейф (например, чекпоинт претрейна).
    """
    names = {id(p): n for n, p in model.named_parameters()}
    per = {}
    for p in split_params(model)["matrix"]:
        name = names[id(p)]
        row = {"stable_rank": stable_rank(p)}
        if ref_state is not None:
            row["drift"] = relative_drift(p, ref_state[name])
        per[name] = row
    report = {
        "per_matrix": per,
        "mean_stable_rank": sum(r["stable_rank"] for r in per.values()) / len(per),
    }
    if ref_state is not None:
        report["mean_drift"] = sum(r["drift"] for r in per.values()) / len(per)
    return report


def _files_hash(root: Path, rels) -> str:
    h = hashlib.sha256()
    for rel in rels:
        h.update(str(rel).encode())
        data = (root / rel).read_bytes()
        h.update(data.replace(b"\r\n", b"\n") if str(rel).endswith(".py") else data)
    return h.hexdigest()


def student_code_hash(root=None) -> str:
    """SHA-256 по содержимому пяти TODO-файлов (имя файла + текст, переводы строк нормализованы)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    return _files_hash(root, TODO_FILES)


def tests_hash(root=None) -> str:
    """SHA-256 по tests/*.py и tests/fixtures/*.pt: грейдер сверяет его с эталонным (тесты менять нельзя)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    rels = sorted(p.relative_to(root).as_posix() for p in (root / "tests").rglob("*") if p.suffix in {".py", ".pt"})
    return _files_hash(root, rels)


def env_info() -> dict:
    return {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def git_commit(root=None) -> str | None:
    """Хеш текущего коммита репозитория (None, если это не git-репозиторий)."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_meta(optimizer, steps: int, tokens: int) -> dict:
    """Поле 'meta' чекпоинта: чем и как получен."""
    try:
        cfg_hash = compute_config_hash()
    except OSError:
        cfg_hash = None
    return {"config_sha256": cfg_hash, "optimizer": optimizer, "steps": steps, "tokens": tokens,
            "reference_commit": git_commit(), "torch_version": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@torch.no_grad()
def expert_load(model, path, seq_len: int = 512, n_batches: int = 8, batch_size: int = 32) -> list:
    """Нагрузка экспертов: средняя по батчам доля токен-слотов, [n_layers][n_experts] (forward по первым батчам файла)."""
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    autocast = torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)
    acc, n = None, 0
    for x, _ in make_eval_loader(path, seq_len, batch_size, n_batches * batch_size):
        with autocast:
            model(x.to(dev))
        load = torch.stack([layer.moe.last_load.float().cpu() for layer in model.layers])
        acc = load if acc is None else acc + load
        n += 1
    model.train(was_training)
    return (acc / n).tolist()
