"""W7 D4: SOLUTION."""
import torch
import torch.nn as nn

import pixel_world as W
from jepa_ref import sigreg
from probes_ref import apply_probe, copy_last_error, mlp_probe, r2, ridge_probe


class Encoder(nn.Module):
    def __init__(self, D=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, D))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, D=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + 2, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, D))

    def forward(self, z, a):
        return z + self.net(torch.cat([z, a / W.AMAX], -1))


def lewm_loss(enc, pred, o0, a, o1, lam, generator=None):
    z0, z1 = enc(o0), enc(o1)
    l_pred = ((pred(z0, a) - z1) ** 2).mean()
    total = l_pred + lam * sigreg(torch.cat([z0, z1]), 64, generator) if lam > 0 else l_pred
    return total, l_pred


def train_lewm(data, lam, steps=1500, batch=128, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    enc, pred = Encoder(), Predictor()
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=lr)
    g = torch.Generator().manual_seed(seed)
    O, A = data["obs"], data["action"]
    E, T = A.shape[:2]
    for _ in range(steps):
        e = torch.randint(0, E, (batch,), generator=g)
        t = torch.randint(0, T, (batch,), generator=g)
        loss, _ = lewm_loss(enc, pred, O[e, t], A[e, t], O[e, t + 1], lam, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc.eval(), pred.eval()


@torch.no_grad()
def latent_report(enc, pred, train, val):
    E, T1 = val["obs"].shape[:2]
    Z = enc(val["obs"].flatten(0, 1)).view(E, T1, -1)
    pe = ((pred(Z[:, :-1].flatten(0, 1), val["action"].flatten(0, 1)) - Z[:, 1:].flatten(0, 1)) ** 2).sum(-1).mean().item()
    ce = copy_last_error(Z)
    Ztr = enc(train["obs"].flatten(0, 1))
    Ytr, Yva, Zva = train["pos"].flatten(0, 1), val["pos"].flatten(0, 1), Z.flatten(0, 1)
    W_, b = ridge_probe(Ztr, Ytr)
    with torch.enable_grad():
        probe = mlp_probe(Ztr, Ytr)
    return {"z_std": Z.std(dim=(0, 1)).mean().item(), "pred_err": pe, "copy_err": ce, "ratio": pe / max(ce, 1e-12),
            "lin_r2": r2(apply_probe(Zva, W_, b), Yva), "mlp_r2": r2(probe(Zva), Yva)}
