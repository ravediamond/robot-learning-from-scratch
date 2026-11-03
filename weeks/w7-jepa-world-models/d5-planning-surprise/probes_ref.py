"""PROVIDED: probes and baselines (the W7 D3 solution)."""
import torch
import torch.nn as nn


def ridge_probe(Z, Y, lam=1e-3):
    X = torch.cat([Z, torch.ones(len(Z), 1, dtype=Z.dtype)], 1)              # (N, d+1)
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=Z.dtype)
    Wb = torch.linalg.solve(A, X.T @ Y)                                      # (d+1, k)
    return Wb[:-1], Wb[-1]


def apply_probe(Z, W, b):
    return Z @ W + b


def r2(pred, Y):
    return (1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()).item()


def mlp_probe(Z, Y, steps=800, seed=0):
    torch.manual_seed(seed)
    Z, Y = Z.detach(), Y.detach()
    net = nn.Sequential(nn.Linear(Z.shape[1], 64), nn.ReLU(), nn.Linear(64, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    with torch.enable_grad():
        for _ in range(steps):
            loss = ((net(Z) - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


def copy_last_error(Z):
    return ((Z[:, 1:] - Z[:, :-1]) ** 2).sum(-1).mean().item()
