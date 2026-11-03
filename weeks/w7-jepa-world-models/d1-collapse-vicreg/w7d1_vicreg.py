"""W7 D1: SOLUTION."""
import math

import torch
import torch.nn as nn


def _var(z, eps):
    return torch.relu(1 - torch.sqrt(z.var(dim=0) + eps)).mean()


def _cov(z):
    N, d = z.shape
    zc = z - z.mean(0)
    C = zc.T @ zc / (N - 1)
    off = C - torch.diag(torch.diag(C))
    return (off ** 2).sum() / d


def vicreg_loss(z1, z2, sim_w=25.0, var_w=25.0, cov_w=1.0, use_cov=True, eps=1e-4):
    loss = sim_w * ((z1 - z2) ** 2).mean() + var_w * (_var(z1, eps) + _var(z2, eps))
    if use_cov:
        loss = loss + cov_w * (_cov(z1) + _cov(z2))
    return loss


def effective_rank(z):
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    p = p[p > 0]
    return math.exp(-(p * p.log()).sum().item())


def train_siamese(data, kind, steps=500, batch=256, seed=0):
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = data[torch.randint(0, len(data), (batch,), generator=g)]
        z1 = enc(x + 0.05 * torch.randn(x.shape, generator=g))
        z2 = enc(x + 0.05 * torch.randn(x.shape, generator=g))
        if kind == "invariance":
            loss = ((z1 - z2) ** 2).mean()
        else:
            loss = vicreg_loss(z1, z2, use_cov=(kind == "vicreg"))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc.eval(), loss.item()
