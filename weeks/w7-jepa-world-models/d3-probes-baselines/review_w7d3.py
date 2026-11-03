"""REVIEW. R² for latent probes. Every probe, even on a random untrained encoder, scores above 0.9.
Find the bug, fix, replace:
BUG:
"""
import torch


def r_squared(pred: torch.Tensor, Y: torch.Tensor) -> float:
    ss_res = ((Y - pred) ** 2).sum()
    ss_tot = ((pred - pred.mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()
