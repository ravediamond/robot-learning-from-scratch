"""W3 D2: SOLUTION."""
import math

import torch
import torch.nn as nn


def split_heads(x, h):
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)            # (B, h, T, Dh)


def merge_heads(x):
    B, h, T, Dh = x.shape
    return x.transpose(1, 2).reshape(B, T, h * Dh)


class MultiHeadAttention(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.h = h
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D) for _ in range(4))
        self.last_attn = None

    def forward(self, x, context=None, mask=None):
        ctx = x if context is None else context
        q = split_heads(self.q_proj(x), self.h)                 # (B, h, Tq, Dh)
        k = split_heads(self.k_proj(ctx), self.h)               # (B, h, Tk, Dh)
        v = split_heads(self.v_proj(ctx), self.h)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        A = torch.nan_to_num(torch.softmax(scores, -1), nan=0.0)
        self.last_attn = A.detach()
        return self.out_proj(merge_heads(A @ v))
