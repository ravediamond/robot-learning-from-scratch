"""W6 D1: SOLUTION."""
import math

import torch
import torch.nn as nn

import sim2d


class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=2, H=8, z_dim=4, hidden=256):
        super().__init__()
        self.H, self.z_dim = H, z_dim
        self.enc = nn.Sequential(nn.Linear(obs_dim + H * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2 * z_dim))
        self.dec = nn.Sequential(nn.Linear(obs_dim + z_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, H * 2))

    def encode(self, obs, chunk):
        return self.enc(torch.cat([obs, chunk.flatten(1)], -1)).chunk(2, -1)

    def decode(self, obs, z):
        return self.dec(torch.cat([obs, z], -1)).view(-1, self.H, 2)

    def forward(self, obs, chunk, eps):
        mu, logvar = self.encode(obs, chunk)
        return self.decode(obs, mu + (0.5 * logvar).exp() * eps), mu, logvar

    @torch.no_grad()
    def act(self, obs, generator=None, sample_z=False):
        z = torch.randn(len(obs), self.z_dim, generator=generator) if sample_z else torch.zeros(len(obs), self.z_dim)
        return self.decode(obs, z) * sim2d.MAX_STEP


def act_loss(pred, target, mask, mu, logvar, beta):
    l1 = ((pred - target).abs().mean(-1) * mask).sum() / mask.sum()
    kl = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(-1).mean()
    return l1 + beta * kl


def train_act(demos, H, steps=3000, batch=256, lr=1e-3, beta=0.01, seed=0):
    torch.manual_seed(seed)
    pol = ACTPolicy(H=H)
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    y, mk = demos["chunks"] / sim2d.MAX_STEP, demos["mask"].float()
    for _ in range(steps):
        i = torch.randint(0, len(y), (batch,), generator=g)
        pred, mu, logvar = pol(demos["obs"][i], y[i], torch.randn(batch, pol.z_dim, generator=g))
        loss = act_loss(pred, y[i], mk[i], mu, logvar, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return pol.eval()


class TemporalEnsembler:
    def __init__(self, H, m=0.1):
        self.H, self.m, self.buf = H, m, []          # list of (step_made, chunk (N, H, D))
        self.t = 0

    def step(self, chunk):
        self.buf.append((self.t, chunk))
        self.buf = [(s, c) for s, c in self.buf if self.t - s < self.H]
        acts = torch.stack([c[:, self.t - s] for s, c in self.buf])          # (K, N, D), oldest first
        w = torch.exp(-self.m * torch.arange(len(acts), dtype=torch.float32))
        w = w / w.sum()
        self.t += 1
        return (w[:, None, None] * acts).sum(0)
