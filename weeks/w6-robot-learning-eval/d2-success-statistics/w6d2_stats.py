"""W6 D2: SOLUTION."""
import math
from statistics import NormalDist

import torch


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(values, stat, B=2000, seed=0, alpha=0.05):
    g = torch.Generator().manual_seed(seed)
    n = len(values)
    stats = torch.stack([torch.as_tensor(stat(values[torch.randint(0, n, (n,), generator=g)]), dtype=torch.float64)
                         for _ in range(B)])
    q = torch.quantile(stats, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64))
    return q[0].item(), q[1].item()


def paired_bootstrap_diff(a, b, B=2000, seed=0):
    d = (a - b).double()
    lo, hi = bootstrap_ci(d, lambda x: x.mean(), B, seed)
    return d.mean().item(), lo, hi


def mcnemar_exact(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def trials_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = NormalDist().inv_cdf(1 - alpha / 2), NormalDist().inv_cdf(power)
    pb = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(num / (p1 - p2) ** 2)
