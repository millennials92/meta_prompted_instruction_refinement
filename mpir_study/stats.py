"""Small statistics helpers: paired bootstrap, hierarchical bootstrap, noise-floor power."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
RNG_SEED = 20261019


def paired_bootstrap(a: Sequence[bool], b: Sequence[bool], resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = RNG_SEED) -> dict:
    """Mean difference b - a (accuracy points as a fraction) with a 95% paired percentile interval.

    a and b are per-example correctness of two prompts on the same examples."""
    x = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    if not np.any(x):
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": n, "discordant": 0, "gained": 0, "lost": 0}
    idx = rng.integers(0, n, size=(resamples, n))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": n,
            "discordant": int(np.count_nonzero(x)), "gained": int((x > 0).sum()), "lost": int((x < 0).sum())}


def is_win(result: dict) -> bool:
    return result["diff"] > 0 and result["lo"] > 0


def hierarchical_bootstrap(values_by_task: dict[str, list[float]], resamples: int = BOOTSTRAP_RESAMPLES,
                           seed: int = RNG_SEED) -> dict:
    """Mean of per-cell differences with a 95% interval, resampling tasks first and cells within a task."""
    tasks = list(values_by_task)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for r in range(resamples):
        picked = rng.integers(0, len(tasks), size=len(tasks))
        pooled = []
        for t in picked:
            cells = values_by_task[tasks[t]]
            pooled.extend(rng.choice(cells, size=len(cells), replace=True))
        means[r] = np.mean(pooled)
    all_values = [v for vs in values_by_task.values() for v in vs]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(np.mean(all_values)), "lo": float(lo), "hi": float(hi), "cells": len(all_values)}


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    from scipy.stats import spearmanr
    if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def minimum_detectable_effect(n: int, discordance: float, alpha_z: float = 1.96, power_z: float = 0.84) -> float:
    """Smallest true paired accuracy difference detectable with ~80% power at alpha 0.05.

    The paired difference has standard error sqrt(discordance / n) when discordant pairs are split evenly
    (the McNemar variance); MDE = (z_alpha + z_power) * SE."""
    return (alpha_z + power_z) * math.sqrt(discordance / n)
