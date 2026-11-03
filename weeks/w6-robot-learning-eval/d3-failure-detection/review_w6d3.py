"""REVIEW. A fast AUROC for failure monitors. Using episode LENGTH as the score, it reports
0.97 on one ordering of the rollout table and 0.62 on another ordering of the SAME table.
Find the bug, fix, replace:
BUG:
"""
import torch


def fast_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    ranks = torch.empty(len(scores))
    ranks[torch.argsort(scores)] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    return ((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
