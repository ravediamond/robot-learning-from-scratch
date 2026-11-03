"""W7 D5: SOLUTION."""
import sys
from pathlib import Path

import torch

import pixel_world as W

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d4-mini-lewm"))
from w7d4_lewm import Encoder, Predictor  # noqa: E402,F401


@torch.no_grad()
def cem_plan(pred, z0, cost_fn, H=5, N=200, K=20, iters=4, generator=None):
    E, D = z0.shape
    mu, std = torch.zeros(E, H, 2), torch.full((E, H, 2), W.AMAX)
    for _ in range(iters):
        a = (mu[:, None] + std[:, None] * torch.randn(E, N, H, 2, generator=generator)).clamp(-W.AMAX, W.AMAX)
        z = z0[:, None].expand(E, N, D).reshape(E * N, D)
        cost = torch.zeros(E, N)
        for h in range(H):
            z = pred(z, a[:, :, h].reshape(E * N, 2))
            cost = cost + cost_fn(z.view(E, N, D))
        idx = cost.topk(K, largest=False).indices                              # (E, K)
        el = torch.gather(a, 1, idx[:, :, None, None].expand(E, K, H, 2))
        mu, std = el.mean(1), el.std(1) + 1e-3
    return mu[:, 0]


def latent_cost(z_goal):
    return lambda z: ((z - z_goal[:, None]) ** 2).sum(-1)


def probe_cost(probe, goal_pos):
    return lambda z: ((probe(z) - goal_pos[:, None]) ** 2).sum(-1)


@torch.no_grad()
def run_planner(enc, pred, start, goal, make_cost, steps=25, seed=0, act_fn=None):
    g = torch.Generator().manual_seed(seed)
    cost = make_cost(enc(W.render(goal)), goal) if make_cost is not None else None
    pos = start.clone()
    for _ in range(steps):
        z = enc(W.render(pos))
        a = act_fn(z, g) if act_fn is not None else cem_plan(pred, z, cost, generator=g)
        pos = W.step(pos, a)
    return pos


@torch.no_grad()
def surprise_scores(enc, pred, n_episodes=50, jump=0.6, seed=3):
    d = W.make_episodes(n_episodes, 20, seed)
    P = d["pos"].clone()
    P[:, 11:] = (P[:, 11:] + jump).clamp(-W.LIM, W.LIM)
    Z = enc(W.render(P.reshape(-1, 2))).view(n_episodes, 21, -1)
    s = ((pred(Z[:, :-1].flatten(0, 1), d["action"].flatten(0, 1)) - Z[:, 1:].flatten(0, 1)) ** 2).sum(-1)
    lab = torch.zeros(n_episodes, 20, dtype=torch.bool)
    lab[:, 10] = True
    return s, lab.flatten()
