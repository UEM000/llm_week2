"""Крошечный синтетический датасет в формате из data/loader.py -- для smoke-прогона на CPU.

Токены: 0 = <eos>, 1 = <pad>, 2..11 = цифры, 12..16 = служебные слова passkey, остальные -- «слова» марковской цепи.
"""
import json
from pathlib import Path

import numpy as np

from kestrel.config import Config

EOS, PAD = 0, 1
DIGITS = list(range(2, 12))
W_THE, W_PASS, W_KEY, W_IS, W_REMEMBER = 12, 13, 14, 15, 16
FIRST_WORD = 17


class _Chain:
    """Марковская цепь: у каждого слова 3 возможных преемника -- лосс заметно падает за десятки шагов."""

    def __init__(self, vocab_size: int, rng: np.random.Generator):
        self.words = np.arange(FIRST_WORD, vocab_size)
        self.next = rng.choice(self.words, size=(vocab_size, 3))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        cur = int(rng.choice(self.words))
        for i in range(n):
            out[i] = cur
            cur = int(self.next[cur, rng.integers(3)])
        return out


def _passkey_example(chain: _Chain, total_len: int, depth: float, rng: np.random.Generator):
    """(prompt_ids, answer_ids), len(prompt) + len(answer) = total_len."""
    digits = rng.choice(DIGITS, size=5).tolist()
    needle = [W_THE, W_PASS, W_KEY, W_IS, *digits, W_REMEMBER]
    question = [W_PASS, W_KEY, W_IS]  # ответ (5 цифр) идёт сразу за вопросом
    n_filler = total_len - len(needle) - len(question) - len(digits)
    filler = chain.sample(n_filler, rng).tolist()
    cut = int(depth * n_filler)
    prompt = filler[:cut] + needle + filler[cut:] + question
    return prompt, digits


def _write(path: Path, arr: np.ndarray) -> None:
    arr.astype(np.uint16).tofile(path)


def make_synthetic_dataset(out_dir, cfg: Config, seed: int = 1337, n_pretrain_windows: int = 400,
                           n_long: int = 120, n_val: int = 16) -> Path:
    """Пишет pretrain_*.bin, long_*.bin, passkey_eval.jsonl, meta.json под размеры из cfg. Идемпотентна.

    pretrain_*.bin -- плоский поток (n * seq_len + 1 токенов => n окон); long_*.bin -- n строк по seq_len токенов,
    в train 10% строк -- passkey-примеры длиной ровно seq_len.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    chain = _Chain(cfg.model.vocab_size, rng)
    s_pre, s_long = cfg.pretrain.seq_len, cfg.finetune.seq_len

    _write(out / "pretrain_train.bin", chain.sample(n_pretrain_windows * s_pre + 1, rng))
    _write(out / "pretrain_val.bin", chain.sample(n_val * s_pre + 1, rng))

    n_pass = max(1, n_long // 10)
    rows = [chain.sample(s_long, rng) for _ in range(n_long - n_pass)]
    for _ in range(n_pass):
        prompt, answer = _passkey_example(chain, s_long, rng.uniform(0.1, 0.9), rng)
        rows.append(np.array(prompt + answer))
    _write(out / "long_train.bin", np.concatenate(rows))
    _write(out / "long_val.bin", np.concatenate([chain.sample(s_long, rng) for _ in range(n_val)]))

    eval_rows = []
    for length in (s_long // 2, s_long):
        for depth in (0.1, 0.3, 0.5, 0.7, 0.9):
            for _ in range(10):
                prompt, answer = _passkey_example(chain, length, depth, rng)
                eval_rows.append({"input_ids": prompt, "answer_ids": answer, "length": length, "depth": depth})
    with open(out / "passkey_eval.jsonl", "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + chr(10))

    (out / "meta.json").write_text(json.dumps({"synthetic": True, "seed": seed}), encoding="utf-8")
    return out
