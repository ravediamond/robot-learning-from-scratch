"""W3 D1: SOLUTION."""
import math

import torch


def causal_mask(T, device=None):
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def key_padding_mask(lengths, T):
    return (torch.arange(T, device=lengths.device)[None, :] < lengths[:, None])[:, None, :]   # (B, 1, T)


def combine_masks(*masks):
    out = None
    for mk in masks:
        if mk is not None:
            out = mk if out is None else (out & mk)
    return out


def attention_weights(q, k, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])          # (..., Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    A = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(A, nan=0.0)                                  # fully masked rows -> 0


def attention(q, k, v, mask=None):
    return attention_weights(q, k, mask) @ v
