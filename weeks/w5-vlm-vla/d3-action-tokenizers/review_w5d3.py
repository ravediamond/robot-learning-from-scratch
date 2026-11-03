"""REVIEW. Detokenizer of an OpenVLA-style policy. The arm always ends a little too low and too
far left: every joint is systematically off by the same small amount. Find the bug, fix, replace:
BUG:
"""
import torch


def detokenize(tokens: torch.Tensor, low: float, high: float, n_bins: int) -> torch.Tensor:
    width = (high - low) / n_bins
    return low + tokens.float() * width
