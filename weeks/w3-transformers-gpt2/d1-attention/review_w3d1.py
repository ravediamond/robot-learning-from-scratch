"""REVIEW. SOLUTION.

BUG: triu with diagonal=0 also masks the diagonal, so token t cannot see itself; token 0 then has no key
at all -> softmax over all -inf -> NaN. The future starts at diagonal=1.
"""
import math

import torch


def causal_self_attention(x):
    T, d = x.shape[1], x.shape[2]
    scores = x @ x.transpose(1, 2) / math.sqrt(d)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, -1) @ x
