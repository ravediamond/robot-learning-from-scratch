"""W3 D1: attention by hand. No F.scaled_dot_product_attention, no nn.MultiheadAttention."""
import math

import torch


def causal_mask(T: int, device=None) -> torch.Tensor:
    """(T, T) bool, True where query i may attend key j (j <= i)."""
    # TODO
    raise NotImplementedError


def key_padding_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """lengths (B,) -> (B, 1, T) bool, True on real keys (index < length)."""
    # TODO
    raise NotImplementedError


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Logical AND of all non-None masks (broadcasting). None if all are None."""
    # TODO
    raise NotImplementedError


def attention_weights(q: torch.Tensor, k: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """q (..., Tq, d), k (..., Tk, d), mask broadcastable to (..., Tq, Tk), True = allowed.
    Returns A (..., Tq, Tk): softmax(q kᵀ / √d) over keys. Rows with no allowed key -> all zeros (no nan)."""
    # TODO
    raise NotImplementedError


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """(..., Tq, d), (..., Tk, d), (..., Tk, dv) -> (..., Tq, dv)."""
    # TODO
    raise NotImplementedError
