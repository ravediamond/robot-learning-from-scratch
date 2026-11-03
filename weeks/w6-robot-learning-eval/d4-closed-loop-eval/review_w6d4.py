"""REVIEW. Compares two policies. The report says B beats A by 6 points; rerun it and A beats B.
Both policies are deterministic. Find the bug, fix, replace:
BUG:
"""
import torch

import sim2d


@torch.no_grad()
def success_rate(act_fn, n: int = 100, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    pos = sim2d.sample_starts(n, torch.Generator().manual_seed(torch.seed()))   # fresh starts each call
    done = torch.zeros(n, dtype=torch.bool)
    ok = torch.zeros(n, dtype=torch.bool)
    for _ in range(sim2d.T_MAX):
        a = sim2d.clip_action(act_fn(pos, g)) * (~done)[:, None]
        pos = pos + a
        hit = sim2d.collides(pos)
        ok |= sim2d.reached(pos) & ~done & ~hit
        done |= hit | ok
    return ok.float().mean().item()
