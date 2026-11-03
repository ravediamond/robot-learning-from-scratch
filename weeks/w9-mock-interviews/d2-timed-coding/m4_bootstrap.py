import math

import torch


def paired_success_diff_ci(a, b, B, seed):
    d = (a - b).double()
    g = torch.Generator().manual_seed(seed)
    n = len(d)
    stats = torch.stack([d[torch.randint(0, n, (n,), generator=g)].mean() for _ in range(B)])
    q = torch.quantile(stats, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return d.mean().item(), q[0].item(), q[1].item()


def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)
