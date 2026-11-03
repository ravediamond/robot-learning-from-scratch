"""REVIEW. SOLUTION.

BUG: the starts come from an unseeded torch.Generator() (non-deterministic seed), not from `seed`: every
call evaluates on different start states, so two policies are never compared on the same starts and
the ranking flips with sampling noise. Seed the starts (and share them across policies).
"""
import torch

import sim2d


@torch.no_grad()
def success_rate(act_fn, n=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = sim2d.sample_starts(n, torch.Generator().manual_seed(seed))
    done = torch.zeros(n, dtype=torch.bool)
    ok = torch.zeros(n, dtype=torch.bool)
    for _ in range(sim2d.T_MAX):
        a = sim2d.clip_action(act_fn(pos, g)) * (~done)[:, None]
        pos = pos + a
        hit = sim2d.collides(pos)
        ok |= sim2d.reached(pos) & ~done & ~hit
        done |= hit | ok
    return ok.float().mean().item()
