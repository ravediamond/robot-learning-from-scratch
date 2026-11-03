"""W9 D4: SOLUTION."""
import torch


def paired_diff_ci(base, method, B=2000, seed=0):
    d = (method - base).double()
    g = torch.Generator().manual_seed(seed)
    n = len(d)
    stats = torch.stack([d[torch.randint(0, n, (n,), generator=g)].mean() for _ in range(B)])
    q = torch.quantile(stats, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return d.mean().item(), q[0].item(), q[1].item()


def verdict(d, lo, hi, pass_line, fail_line, confounds=None):
    if confounds:
        return "not proven"
    if d >= pass_line and lo > 0:
        return "pass"
    if d <= fail_line and hi < pass_line:
        return "fail"
    return "not proven"
