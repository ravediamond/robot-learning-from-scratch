"""W4 D3: SOLUTION."""
import torch
import torch.nn as nn

from toy2d import time_embedding


def fm_interpolate(x0, x1, t):
    tt = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - tt) * x0 + tt * x1, x1 - x0


class VelocityNet(nn.Module):
    def __init__(self, D, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, D))

    def forward(self, x, t):
        return self.net(torch.cat([x, time_embedding(t, 32)], -1))


def fm_loss(model, x1, generator):
    x0 = torch.randn(x1.shape, generator=generator)
    t = torch.rand(len(x1), generator=generator)
    xt, target = fm_interpolate(x0, x1, t)
    return ((model(xt, t) - target) ** 2).mean()


@torch.no_grad()
def euler_sample(model, n, steps, generator, x0=None):
    x = torch.randn(n, model.net[-1].out_features, generator=generator) if x0 is None else x0.clone()
    for i in range(steps):
        t = torch.full((len(x),), i / steps)
        x = x + model(x, t) / steps
    return x


def train_fm(x, steps=4000, batch=256, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    model = VelocityNet(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch,), generator=g)
        loss = fm_loss(model, x[idx], g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()
