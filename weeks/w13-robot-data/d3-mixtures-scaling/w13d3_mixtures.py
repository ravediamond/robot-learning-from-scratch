"""W13 D3: mixture weights, sampling, scaling-law fit, seeded ablations."""
import math

import numpy as np


def temperature_weights(sizes: dict[str, float], T: float) -> dict[str, float]:
    """p_i proportional to sizes[i] ** (1 / T), normalised to sum to 1."""
    raw = {k: v ** (1.0 / T) for k, v in sizes.items()}
    z = sum(raw.values())
    return {k: v / z for k, v in raw.items()}


def item_weights(source_of_item: list[str], p_source: dict[str, float]) -> np.ndarray:
    """Per-item sampling weights (summing to 1) so that a sampler drawing items with these weights picks
    source s with probability p_source[s]: weight = p_source[s] / (number of items of s)."""
    counts: dict[str, int] = {}
    for s in source_of_item:
        counts[s] = counts.get(s, 0) + 1
    w = np.array([p_source[s] / counts[s] for s in source_of_item], dtype=float)
    return w / w.sum()


def epochs_per_source(sizes: dict[str, float], p: dict[str, float], budget: float) -> dict[str, float]:
    """Expected passes over each source when `budget` units (same unit as sizes) are sampled: budget * p / size."""
    return {k: budget * p[k] / sizes[k] for k in sizes}


def fit_power_law(n: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of log(err) = log(a) - b log(n). Return (a, b)."""
    slope, intercept = np.polyfit(np.log(n), np.log(err), 1)
    return float(math.exp(intercept)), float(-slope)


def ablation_effect(with_x: list[float], without_x: list[float], z: float = 1.96) -> dict:
    """Paired by seed (same seeds, same order). d_i = with_x[i] - without_x[i].
    Return {"effect": mean(d), "se": sample sd(d) / sqrt(k) (ddof = 1), "lo": effect - z*se, "hi": effect + z*se,
    "unpaired_se": sqrt(var(with)/k + var(without)/k) (ddof = 1)} with every value rounded to 4 decimals."""
    a, b = np.asarray(with_x, float), np.asarray(without_x, float)
    d = a - b
    k = len(d)
    se = d.std(ddof=1) / math.sqrt(k)
    unpaired = math.sqrt(a.var(ddof=1) / k + b.var(ddof=1) / k)
    e = d.mean()
    return {k_: round(float(v), 4) for k_, v in
            {"effect": e, "se": se, "lo": e - z * se, "hi": e + z * se, "unpaired_se": unpaired}.items()}
