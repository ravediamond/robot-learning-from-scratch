"""REVIEW. Token merging for a VLM (4 neighbouring patch tokens -> 1). Shapes are right; the
VLA's localisation got much worse after adding it, more than expected from the lower resolution.
Find the bug, fix, replace:
BUG:
"""
import torch


def merge_tokens(x: torch.Tensor, r: int = 2) -> torch.Tensor:
    """(B, h*h, C) -> (B, (h/r)^2, C*r*r), merging r x r spatial neighbours."""
    B, N, C = x.shape
    return x.reshape(B, N // (r * r), C * r * r)
