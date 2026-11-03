"""W6 D2: statistics for robot evaluation. torch + the standard library only."""
import math
from statistics import NormalDist
from typing import Callable

import torch


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval, clipped to [0, 1]. n = 0 -> (0.0, 1.0)."""
    # TODO
    raise NotImplementedError


def bootstrap_ci(values: torch.Tensor, stat: Callable[[torch.Tensor], torch.Tensor], B: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """values (n, ...): one row per EPISODE. Resample rows with replacement B times (torch.Generator(seed)),
    compute stat on each resample, return the (alpha/2, 1-alpha/2) percentiles (torch.quantile)."""
    # TODO
    raise NotImplementedError


def paired_bootstrap_diff(a: torch.Tensor, b: torch.Tensor, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """a, b (n,) outcomes of two policies on the SAME n start states. Returns (mean(a - b), lo, hi) with the
    95% percentile bootstrap over start states."""
    # TODO
    raise NotImplementedError


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    # TODO
    raise NotImplementedError


def trials_needed(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Trials PER ARM (rounded up) to distinguish p1 from p2, two independent groups, two-sided test."""
    # TODO
    raise NotImplementedError
