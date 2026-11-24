import math

import torch


def yarn_inv_freq(inv_freq: torch.Tensor, scale: float, l_train: int, alpha: float, beta: float) -> torch.Tensor:
    """YaRN-интерполяция частот RoPE для контекста в `scale` раз длиннее обучающего.

    inv_freq: частоты θ формы [Dh/2] (их возвращает `rope_inv_freq`); scale = L_new / l_train;
    alpha, beta -- границы рампы. Возвращает тензор той же формы и dtype (все вычисления в dtype
    входа, сам `inv_freq` не изменяется). При scale == 1.0 нужно вернуть `inv_freq` без изменений:
    тест проверяет точное равенство.

    Формулы: TECH_REPORT.md, раздел 2.3; заметки по реализации: приложение A.1.
    """
    if abs(scale - 1) <= 1e-9:
        return inv_freq.clone()
    r = inv_freq * l_train / (2 * math.pi)
    small_freq_mask = r < alpha
    middle_freq_mask = (r >= alpha) & (r <= beta)

    new_freq = inv_freq.clone()
    new_freq[small_freq_mask] = new_freq[small_freq_mask] / scale
    gammas = (r[middle_freq_mask] - alpha) / (beta - alpha)
    new_freq[middle_freq_mask] = (
        (1 -  gammas) * new_freq[middle_freq_mask] / scale + gammas * new_freq[middle_freq_mask]
    )

    return new_freq



def yarn_mscale(scale: float) -> float:
    """Температурный множитель внимания YaRN (python float); при scale == 1.0 ровно 1.0.

    Формула: TECH_REPORT.md, раздел 2.3; приложение A.1.
    """
    if scale <= 1:
        return 1
    return 0.1 * math.log(scale) + 1
