"""W3 D2: multi-head attention. No nn.MultiheadAttention, no F.scaled_dot_product_attention."""
import math

import torch
import torch.nn as nn


def split_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    """(B, T, D) -> (B, h, T, D/h)."""
    # TODO
    raise NotImplementedError


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, h, T, Dh) -> (B, T, h*Dh)."""
    # TODO
    raise NotImplementedError


class MultiHeadAttention(nn.Module):
    """Attributes q_proj, k_proj, v_proj, out_proj: nn.Linear(D, D) with bias.

    forward(x, context=None, mask=None):
        x (B, Tq, D); context (B, Tk, D) or None (self-attention: context = x)
        mask: bool broadcastable to (B, 1, Tq, Tk) or (Tq, Tk), True = may attend
        returns (B, Tq, D)
    Also store the last attention weights (B, h, Tq, Tk) in self.last_attn (detached).
    """

    def __init__(self, D: int, h: int):
        super().__init__()
        # TODO

    def forward(self, x, context=None, mask=None):
        # TODO
        raise NotImplementedError
