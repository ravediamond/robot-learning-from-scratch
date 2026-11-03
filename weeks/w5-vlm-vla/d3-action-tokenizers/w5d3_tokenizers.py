"""W5 D3: SOLUTION."""
import math

import torch


def bin_encode(a, low, high, n_bins):
    x = (a.clamp(low, high) - low) / (high - low) * n_bins
    return x.floor().long().clamp(0, n_bins - 1)


def bin_decode(tokens, low, high, n_bins):
    width = (high - low) / n_bins
    return low + (tokens.float() + 0.5) * width


def dct_matrix(H):
    n = torch.arange(H, dtype=torch.float32)
    C = torch.cos(math.pi * (n[None, :] + 0.5) * n[:, None] / H)       # (k, n)
    s = torch.full((H,), math.sqrt(2.0 / H))
    s[0] = math.sqrt(1.0 / H)
    return s[:, None] * C


def dct_encode(chunk, keep, scale):
    coeff = dct_matrix(chunk.shape[0]) @ chunk                         # (H, D)
    return torch.round(coeff[:keep] * scale).long()


def dct_decode(tokens, H, scale):
    keep = tokens.shape[0]
    return dct_matrix(H)[:keep].T @ (tokens.float() / scale)
