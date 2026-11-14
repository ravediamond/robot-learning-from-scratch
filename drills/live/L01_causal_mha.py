"""L01 (45 min). Causal multi-head self-attention, with an optional KV cache for incremental decoding."""
import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """x (B, T, C) -> (B, T, C). C = n_heads * head_dim. Linear q, k, v, out projections (with bias), named
    self.q, self.k, self.v, self.out. Scaled dot-product attention with a causal mask (position t attends to
    positions <= t). Do not use nn.MultiheadAttention or F.scaled_dot_product_attention.

    forward(x, cache=None): if cache is a dict {"k": (B, H, S, D), "v": (B, H, S, D)} of PAST keys and values,
    the new tokens x are positions S..S+T-1: they attend to all cached positions and causally to each other.
    Return (output, new_cache) where new_cache holds all keys and values so far."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.d = n_heads, dim // n_heads
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x: torch.Tensor, cache: dict | None = None):
        B, T, C = x.shape

        def heads(t):
            return t.view(B, T, self.h, self.d).transpose(1, 2)        # (B, H, T, D)

        q, k, v = heads(self.q(x)), heads(self.k(x)), heads(self.v(x))
        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        S = k.shape[2] - T                                                 # number of past positions
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)               # (B, H, T, S + T)
        qpos = torch.arange(T, device=x.device)[:, None] + S
        kpos = torch.arange(S + T, device=x.device)[None, :]
        scores = scores.masked_fill(kpos > qpos, float("-inf"))
        y = scores.softmax(dim=-1) @ v                                     # (B, H, T, D)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y), {"k": k, "v": v}
