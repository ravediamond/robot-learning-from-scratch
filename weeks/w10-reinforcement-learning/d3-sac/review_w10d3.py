"""REVIEW. SOLUTION.

BUG: the change-of-variables term is missing. a = tanh(u) squeezes probability mass near +-1, so
log pi(a) = log N(u) - sum log(1 - tanh(u)^2). Without it the function returns the density of u, not of a.
"""
import math

import torch


def squashed_logp(mean: torch.Tensor, log_std: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    z = (u - mean) / log_std.exp()
    log_n = -0.5 * z.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
    return (log_n - torch.log(1 - torch.tanh(u).pow(2) + 1e-6)).sum(dim=-1)
