"""REVIEW. SOLUTION.

BUG: dim=-1 averages over the joints of each sample (one number per sample, shape (N,)), not over
samples for each joint. On a square batch (N == D) the shape is still (D,), so nothing crashes.
"""
import torch


def joint_stats(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = batch.mean(dim=0)  # (D,)
    std = batch.std(dim=0)    # (D,)
    return mean, std
