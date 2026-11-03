"""PROVIDED building blocks for the mini VLA: your W3 multi-head attention and pre-norm Block
(cross-attention with context=...), and the W4 time embedding. Same code as those solutions."""
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


class Block(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = MultiHeadAttention(D, h)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask=None, context=None):
        x = x + self.attn(self.ln_1(x), context=context, mask=mask)
        return x + self.mlp(self.ln_2(x))


def time_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """t (N,) float in [0, 1] (or step/T) -> (N, dim) sinusoidal features."""
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), dim // 2))
    ang = t[:, None] * freqs[None] * 2 * math.pi
    return torch.cat([ang.sin(), ang.cos()], -1)
