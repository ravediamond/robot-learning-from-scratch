"""W13 D3: mixture weights, sampling, scaling-law fit, seeded ablations."""
import math

import numpy as np


def temperature_weights(sizes: dict[str, float], T: float) -> dict[str, float]:
    """p_i proportional to sizes[i] ** (1 / T), normalised to sum to 1."""
    # TODO
    raise NotImplementedError


def item_weights(source_of_item: list[str], p_source: dict[str, float]) -> np.ndarray:
    """Per-item sampling weights (summing to 1) so that a sampler drawing items with these weights picks
    source s with probability p_source[s]: weight = p_source[s] / (number of items of s)."""
    # TODO
    raise NotImplementedError


def epochs_per_source(sizes: dict[str, float], p: dict[str, float], budget: float) -> dict[str, float]:
    """Expected passes over each source when `budget` units (same unit as sizes) are sampled: budget * p / size."""
    # TODO
    raise NotImplementedError


def fit_power_law(n: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of log(err) = log(a) - b log(n). Return (a, b)."""
    # TODO
    raise NotImplementedError


def ablation_effect(with_x: list[float], without_x: list[float], z: float = 1.96) -> dict:
    """Paired by seed (same seeds, same order). d_i = with_x[i] - without_x[i].
    Return {"effect": mean(d), "se": sample sd(d) / sqrt(k) (ddof = 1), "lo": effect - z*se, "hi": effect + z*se,
    "unpaired_se": sqrt(var(with)/k + var(without)/k) (ddof = 1)} with every value rounded to 4 decimals."""
    # TODO
    raise NotImplementedError
