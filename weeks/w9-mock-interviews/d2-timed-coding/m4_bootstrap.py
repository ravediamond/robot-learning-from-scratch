# time:
import torch


def paired_success_diff_ci(a: torch.Tensor, b: torch.Tensor, B: int, seed: int) -> tuple[float, float, float]:
    """a, b (n,) 0/1 outcomes on the same n starts. Returns (mean(a - b), lo, hi) 95% percentile bootstrap
    over starts with torch.Generator(seed)."""
    raise NotImplementedError


def wilson(k: int, n: int) -> tuple[float, float]:
    """z = 1.96."""
    raise NotImplementedError
