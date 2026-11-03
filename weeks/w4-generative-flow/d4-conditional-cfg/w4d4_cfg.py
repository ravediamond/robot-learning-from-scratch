"""W4 D4: SOLUTION."""
import torch
import torch.nn as nn

from toy2d import time_embedding


class CondVelocityNet(nn.Module):
    def __init__(self, D, n_classes, hidden=128):
        super().__init__()
        self.null = n_classes
        self.embed = nn.Embedding(n_classes + 1, 32)
        self.net = nn.Sequential(nn.Linear(D + 64, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, D))

    def forward(self, x, t, c):
        return self.net(torch.cat([x, time_embedding(t, 32), self.embed(c)], -1))


def drop_labels(c, p, null, generator):
    drop = torch.rand(c.shape, generator=generator) < p
    return torch.where(drop, torch.full_like(c, null), c)


def cond_fm_loss(model, x1, c, p_uncond, generator):
    x0 = torch.randn(x1.shape, generator=generator)
    t = torch.rand(len(x1), generator=generator)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    c = drop_labels(c, p_uncond, model.null, generator)
    return ((model(xt, t, c) - (x1 - x0)) ** 2).mean()


def cfg_velocity(model, x, t, c, w):
    v_c = model(x, t, c)
    v_u = model(x, t, torch.full_like(c, model.null))
    return v_u + w * (v_c - v_u)


@torch.no_grad()
def cfg_sample(model, c, steps, w, generator):
    x = torch.randn(len(c), model.net[-1].out_features, generator=generator)
    for i in range(steps):
        x = x + cfg_velocity(model, x, torch.full((len(c),), i / steps), c, w) / steps
    return x


def train_cond_fm(x, c, steps=4000, batch=256, lr=2e-3, p_uncond=0.1, seed=0):
    torch.manual_seed(seed)
    model = CondVelocityNet(x.shape[1], int(c.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), generator=g)
        loss = cond_fm_loss(model, x[idx], c[idx], p_uncond, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()
