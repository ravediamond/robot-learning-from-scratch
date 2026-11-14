"""REVIEW. Log-probability of a tanh-squashed Gaussian action, from the SAC actor. The entropy estimate is
wrong near the action bounds: the temperature keeps rising and the policy saturates at +-1.
Find the bug, fix, replace:
BUG:
"""
import math

import torch


def squashed_logp(mean: torch.Tensor, log_std: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """u = pre-tanh sample, shape (B, A). Returns log pi(tanh(u)), shape (B,)."""
    z = (u - mean) / log_std.exp()
    log_n = -0.5 * z.pow(2) - log_std - 0.5 * math.log(2 * math.pi)
    return log_n.sum(dim=-1)
