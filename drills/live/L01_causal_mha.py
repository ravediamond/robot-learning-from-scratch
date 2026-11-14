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
        # TODO
        raise NotImplementedError
