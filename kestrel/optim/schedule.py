def wsd_lr(step: int, peak: float, total_steps: int, warmup_steps: int, decay_frac: float, final_lr_frac: float) -> float:
    """Warmup-Stable-Decay. `step` -- номер шага, который сейчас будет выполнен (с 0).

    Линейный рост от 0 за warmup_steps, плато на peak, затем линейный спад на последних
    decay_frac*total_steps шагах до final_lr_frac*peak (на последнем шаге lr ровно final_lr_frac*peak).
    """
    decay_steps = int(decay_frac * total_steps)
    decay_start = total_steps - decay_steps
    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    if decay_steps > 0 and step >= decay_start:
        progress = min(1.0, (step - decay_start + 1) / decay_steps)
        return peak * (1.0 - (1.0 - final_lr_frac) * progress)
    return peak
