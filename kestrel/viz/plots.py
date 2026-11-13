"""Графики для отчёта. Все функции принимают готовые данные, рисуют и (если задан out_path) сохраняют PNG.

Бэкенд matplotlib не форсируем: в ноутбуке графики показываются inline, в скриптах без экрана работает Agg
(при необходимости задайте MPLBACKEND=Agg)."""
import matplotlib.pyplot as plt
import numpy as np
import torch


def _finish(fig, out_path):
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=130)
    return fig


def _series(rows: list[dict], key: str):
    pts = [(r["step"], r[key]) for r in rows if key in r]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_loss_curves(logs: dict, key: str = "val_loss", title: str = "", out_path=None):
    """Кривые лосса нескольких прогонов на одной оси. logs: {подпись: список строк лога из read_log}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rows in logs.items():
        xs, ys = _series(rows, key)
        ax.plot(xs, ys, label=label, marker="o" if key == "val_loss" and len(xs) < 40 else None, ms=3)
    ax.set_xlabel("шаг")
    ax.set_ylabel("val loss" if key == "val_loss" else "train loss")
    ax.set_title(title or ("Validation loss" if key == "val_loss" else "Train loss"))
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, out_path)


def plot_finetune_first_steps(logs: dict, n_steps: int = 50, out_path=None):
    """Train loss на первых n_steps шагах дообучения (по шагам, все прогоны на одной оси)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, rows in logs.items():
        xs, ys = _series([r for r in rows if r["step"] <= n_steps], "train_loss")
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("шаг дообучения")
    ax.set_ylabel("train loss")
    ax.set_title(f"Первые {n_steps} шагов дообучения")
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, out_path)


def plot_position_buckets(results: dict, title: str = "Лосс по позициям в контексте", out_path=None):
    """results: {подпись: {'0-511': loss, '512-1023': loss, ...}} -- сгруппированные столбики."""
    labels = list(next(iter(results.values())).keys())
    n = len(results)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, buckets) in enumerate(results.items()):
        ax.bar(np.arange(len(labels)) + i * width, [buckets[k] for k in labels], width, label=name)
    ax.set_xticks(np.arange(len(labels)) + width * (n - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("позиции токенов в куске")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _finish(fig, out_path)


def plot_singular_values(matrices: dict, title: str = "Спектр сингулярных чисел", out_path=None):
    """matrices: {подпись: 2D-тензор}. Сингулярные числа нормированы на максимальное."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, W in matrices.items():
        s = torch.linalg.svdvals(W.detach().float().cpu())
        ax.plot(np.arange(1, len(s) + 1), (s / s[0]).numpy(), label=label)
    ax.set_yscale("log")
    ax.set_xlabel("номер сингулярного числа")
    ax.set_ylabel("σ_i / σ_1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, out_path)


def plot_expert_load(load, title: str = "Нагрузка экспертов", out_path=None):
    """load: [n_layers][n_experts] -- доля токен-слотов (из поля moe_load лога). Тепловая карта слой x эксперт."""
    arr = np.asarray(load, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.set_xlabel("эксперт")
    ax.set_ylabel("слой")
    ax.set_title(f"{title} (идеал = {1 / arr.shape[1]:.3f})")
    fig.colorbar(im, ax=ax)
    return _finish(fig, out_path)
