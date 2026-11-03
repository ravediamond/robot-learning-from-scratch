"""W5 D3: action tokenizers."""
import math

import torch


def bin_encode(a: torch.Tensor, low: float, high: float, n_bins: int) -> torch.Tensor:
    """Any shape -> long tokens in [0, n_bins-1] (values outside [low, high] are clamped)."""
    # TODO
    raise NotImplementedError


def bin_decode(tokens: torch.Tensor, low: float, high: float, n_bins: int) -> torch.Tensor:
    """Tokens -> the CENTRE of each bin (float)."""
    # TODO
    raise NotImplementedError


def dct_matrix(H: int) -> torch.Tensor:
    """(H, H) orthonormal DCT-II matrix (see README)."""
    # TODO
    raise NotImplementedError


def dct_encode(chunk: torch.Tensor, keep: int, scale: float) -> torch.Tensor:
    """chunk (H, D) -> long tokens (keep, D) = round(scale * (C @ chunk)[:keep])."""
    # TODO
    raise NotImplementedError


def dct_decode(tokens: torch.Tensor, H: int, scale: float) -> torch.Tensor:
    """tokens (keep, D) -> chunk (H, D)."""
    # TODO
    raise NotImplementedError
