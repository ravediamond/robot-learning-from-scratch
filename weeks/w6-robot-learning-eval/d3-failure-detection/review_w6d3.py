"""REVIEW. SOLUTION.

BUG: argsort gives tied scores consecutive ranks in whatever order they appear, so with many ties
(episode lengths!) the result depends on row order. Ties must share their average rank (= count 1/2 per tied pair).
"""
import torch


def fast_auroc(scores, labels):
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    for v in torch.unique(scores):
        tie = scores == v
        ranks[tie] = ranks[tie].mean()
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    return ((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()
