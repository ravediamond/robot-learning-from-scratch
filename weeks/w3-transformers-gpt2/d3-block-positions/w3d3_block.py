"""W3 D3: transformer block, positions, patches."""
import math

import torch
import torch.nn as nn

from mha_ref import MultiHeadAttention


class Block(nn.Module):
    """Pre-norm transformer block. Attributes: ln_1, attn (MultiHeadAttention), ln_2,
    mlp = nn.Sequential(Linear(D, 4D), GELU(), Linear(4D, D)).
    forward(x, mask=None, context=None): self-attention if context is None, else cross-attention."""

    def __init__(self, D: int, h: int):
        super().__init__()
        # TODO

    def forward(self, x, mask=None, context=None):
        # TODO
        raise NotImplementedError


def sinusoidal_positions(T: int, D: int) -> torch.Tensor:
    """(T, D): even dims sin(t / 10000^(2i/D)), odd dims cos(same)."""
    # TODO
    raise NotImplementedError


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x (..., T, d) with even d, positions (T,) int or float.
    Rotate each pair (x[..., 2i], x[..., 2i+1]) by angle positions[t] * base**(-2i/d)."""
    # TODO
    raise NotImplementedError


def patchify(img: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/p)*(W/p), C*p*p). Patches in row-major order; each patch flattened
    in (C, p, p) order. H and W divisible by p."""
    # TODO
    raise NotImplementedError
