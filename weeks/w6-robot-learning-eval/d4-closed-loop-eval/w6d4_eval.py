"""W6 D4: SOLUTION."""
import torch

import sim2d


def fixed_starts(n, seed):
    return sim2d.sample_starts(n, torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(act_fn, starts, H_exec, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = starts.clone()
    n = len(pos)
    success = torch.zeros(n, dtype=torch.bool)
    collision = torch.zeros(n, dtype=torch.bool)
    steps = torch.zeros(n, dtype=torch.long)
    t = 0
    while t < sim2d.T_MAX and not (success | collision).all():
        idx = (~(success | collision)).nonzero().squeeze(1)
        chunk = act_fn(pos[idx], g)
        for k in range(min(H_exec, chunk.shape[1])):
            live = ~(success[idx] | collision[idx])
            pos[idx] = pos[idx] + sim2d.clip_action(chunk[:, k]) * live[:, None]
            steps[idx] += live.long()
            collision[idx] |= sim2d.collides(pos[idx]) & live
            success[idx] |= sim2d.reached(pos[idx]) & live & ~collision[idx]
            t += 1
            if t >= sim2d.T_MAX:
                break
    return {"success": success, "collision": collision, "steps": steps}


def paired_counts(a, b):
    a, b = a.bool(), b.bool()
    return {"n11": int((a & b).sum()), "n10": int((a & ~b).sum()), "n01": int((~a & b).sum()), "n00": int((~a & ~b).sum())}


@torch.no_grad()
def val_chunk_mse(act_fn, demos, seed=0):
    pred = act_fn(demos["obs"], torch.Generator().manual_seed(seed))
    err = ((pred - demos["chunks"]) ** 2).mean(-1)
    m = demos["mask"].float()
    return ((err * m).sum() / m.sum()).item()
