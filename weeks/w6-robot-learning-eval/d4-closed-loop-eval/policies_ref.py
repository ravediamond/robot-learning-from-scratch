"""PROVIDED reference policies and rollout (the W4 D5 solution)."""
import torch
import torch.nn as nn

import sim2d
from toy2d import time_embedding


class MSEChunkPolicy(nn.Module):
    def __init__(self, H, hidden=256):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, H * 2))

    def forward(self, obs):
        return self.net(obs).view(-1, self.H, 2)                     # scaled actions (units of MAX_STEP)

    @torch.no_grad()
    def act(self, obs, generator=None):
        return self(obs) * sim2d.MAX_STEP


class FlowChunkPolicy(nn.Module):
    def __init__(self, H, hidden=256):
        super().__init__()
        self.H = H
        self.net = nn.Sequential(nn.Linear(H * 2 + 32 + 2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, H * 2))

    def forward(self, x, t, obs):
        B = len(x)
        return self.net(torch.cat([x.reshape(B, -1), time_embedding(t, 32), obs], -1)).view(B, self.H, 2)

    @torch.no_grad()
    def act(self, obs, generator=None, steps=10):
        x = torch.randn(len(obs), self.H, 2, generator=generator)
        for i in range(steps):
            x = x + self(x, torch.full((len(obs),), i / steps), obs) / steps
        return x * sim2d.MAX_STEP


def masked_mse(pred, target, mask):
    err = ((pred - target) ** 2).mean(-1)
    return (err * mask).sum() / mask.sum()


def train_mse_policy(demos, H, steps=3000, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    pol = MSEChunkPolicy(H)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    y = demos["chunks"] / sim2d.MAX_STEP
    m = demos["mask"].float()
    for _ in range(steps):
        i = torch.randint(0, len(y), (batch,), generator=g)
        loss = masked_mse(pol(demos["obs"][i]), y[i], m[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol.eval()


def train_flow_policy(demos, H, steps=4000, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    pol = FlowChunkPolicy(H)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    y = demos["chunks"] / sim2d.MAX_STEP
    m = demos["mask"].float()
    for _ in range(steps):
        i = torch.randint(0, len(y), (batch,), generator=g)
        x1 = y[i]
        x0 = torch.randn(x1.shape, generator=g)
        t = torch.rand(batch, generator=g)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        loss = masked_mse(pol(xt, t, demos["obs"][i]), x1 - x0, m[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol.eval()


@torch.no_grad()
def rollout(act_fn, n: int, H_exec: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    pos = sim2d.sample_starts(n, g)
    success = torch.zeros(n, dtype=torch.bool)
    collision = torch.zeros(n, dtype=torch.bool)
    steps = torch.zeros(n, dtype=torch.long)
    t = 0
    while t < sim2d.T_MAX:
        active = ~(success | collision)
        if not active.any():
            break
        chunk = act_fn(pos[active], g)                                    # (n_active, H, 2)
        for k in range(min(H_exec, chunk.shape[1])):
            idx = active.nonzero().squeeze(1)
            live = ~(success[idx] | collision[idx])
            a = sim2d.clip_action(chunk[:, k]) * live[:, None]
            pos[idx] = pos[idx] + a
            steps[idx] += live.long()
            collision[idx] |= sim2d.collides(pos[idx]) & live
            success[idx] |= sim2d.reached(pos[idx]) & live & ~collision[idx]
            t += 1
            if t >= sim2d.T_MAX:
                break
    return {"success": success, "collision": collision, "steps": steps}
