"""W4 D2: SOLUTION."""
import torch
import torch.nn as nn

from toy2d import time_embedding


def linear_betas(T, b0=1e-4, b1=0.02):
    return torch.linspace(b0, b1, T)


def alphas_bar(betas):
    return torch.cumprod(1 - betas, dim=0)


def q_sample(x0, t, noise, abar):
    a = abar[t][:, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


class Denoiser(nn.Module):
    def __init__(self, D, T, hidden=128):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(nn.Linear(D + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, D))

    def forward(self, x, t):
        return self.net(torch.cat([x, time_embedding(t.float() / self.T, 32)], -1))


def ddpm_loss(model, x0, abar, generator):
    t = torch.randint(0, len(abar), (len(x0),), generator=generator)
    noise = torch.randn(x0.shape, generator=generator)
    return ((model(q_sample(x0, t, noise, abar), t) - noise) ** 2).mean()


@torch.no_grad()
def sample(model, n, betas, generator):
    abar = alphas_bar(betas)
    x = torch.randn(n, model.net[-1].out_features, generator=generator)
    for t in reversed(range(len(betas))):
        tt = torch.full((n,), t)
        eps = model(x, tt)
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / (1 - betas[t]).sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn(x.shape, generator=generator)
    return x


def train_ddpm(x, T=100, steps=4000, batch=256, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    betas = linear_betas(T)
    abar = alphas_bar(betas)
    model = Denoiser(x.shape[1], T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), generator=g)
        loss = ddpm_loss(model, x[idx], abar, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval(), betas
