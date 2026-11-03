"""REVIEW. The variance term of a VICReg-style regulariser for robot frame embeddings.
Embeddings still collapse (every frame maps to nearly the same vector) although the term is on.
Find the bug, fix, replace:
BUG:
"""
import torch


def variance_term(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """z (N, d): penalise dimensions whose std across the batch is below 1."""
    std = torch.sqrt(z.var(dim=1) + eps)
    return torch.relu(1 - std).mean()
