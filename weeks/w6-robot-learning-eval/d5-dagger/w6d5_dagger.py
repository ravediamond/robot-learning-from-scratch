"""W6 D5: SOLUTION."""
import torch
import torch.nn as nn

import dagger_world as w
import sim2d


def train_bc(obs, act, steps=1500, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    y = act / sim2d.MAX_STEP
    for _ in range(steps):
        i = torch.randint(0, len(obs), (256,), generator=g)
        loss = ((net(obs[i]) - y[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def run(policy, n, seed, record=False):
    g = torch.Generator().manual_seed(seed)
    pos = sim2d.sample_starts(n, g)
    kick = w.pushes(n, g)
    succ = torch.zeros(n, dtype=torch.bool)
    col = torch.zeros(n, dtype=torch.bool)
    visited = []
    for t in range(sim2d.T_MAX):
        live = ~(succ | col)
        if t == w.KICK_T:
            pos = pos + kick * live[:, None]
        if record:
            visited.append(pos[live].clone())
        pos = pos + sim2d.clip_action(policy(pos) * sim2d.MAX_STEP) * live[:, None]
        col |= sim2d.collides(pos) & live
        succ |= sim2d.reached(pos) & live & ~col
    out = {"success": succ, "collision": col}
    if record:
        out["visited"] = torch.cat(visited)
    return out


def aggregate(obs, act, visited):
    return torch.cat([obs, visited]), torch.cat([act, w.expert(visited)])


def dagger(obs, act, iterations, n_rollouts=50, seed=100):
    pol = train_bc(obs, act)
    for i in range(iterations):
        obs, act = aggregate(obs, act, run(pol, n_rollouts, seed + i, record=True)["visited"])
        pol = train_bc(obs, act, seed=i + 1)
    return pol, len(obs)
