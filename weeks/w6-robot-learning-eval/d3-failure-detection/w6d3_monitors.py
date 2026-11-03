"""W6 D3: SOLUTION."""
import math

import torch


def _avg_ranks(x):
    order = torch.argsort(x)
    ranks = torch.empty(len(x), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(x) + 1, dtype=torch.float64)
    for v in torch.unique(x):                                   # ties -> average rank
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def auroc(scores, labels):
    labels = labels.long()
    r = _avg_ranks(scores.double())
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    return ((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


def auroc_ci(scores, labels, B=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(scores)
    vals = []
    for _ in range(B):
        i = torch.randint(0, n, (n,), generator=g)
        if 0 < labels[i].sum() < n:
            vals.append(auroc(scores[i], labels[i]))
    q = torch.quantile(torch.tensor(vals, dtype=torch.float64), torch.tensor([0.025, 0.975], dtype=torch.float64))
    return q[0].item(), q[1].item()


def tpr_at_fpr(scores, labels, fpr):
    thr = torch.quantile(scores[labels == 0].double(), 1 - fpr)
    return (scores[labels == 1].double() > thr).double().mean().item()


def conformal_threshold(calib_success_scores, alpha):
    n = len(calib_success_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return torch.sort(calib_success_scores).values[k - 1].item()
