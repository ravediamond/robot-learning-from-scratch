"""REVIEW. A causal decoder produced NaN loss at step 0. The author added a "no future" mask.
Find the bug, fix, replace:
BUG:
"""
import math

import torch


def causal_self_attention(x: torch.Tensor) -> torch.Tensor:
    """x (B, T, d), q = k = v = x, causal: token t attends tokens 0..t."""
    T, d = x.shape[1], x.shape[2]
    scores = x @ x.transpose(1, 2) / math.sqrt(d)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=0)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, -1) @ x
