"""W7 D2: SOLUTION."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def epps_pulley(p, T=3.0, K=17):
    t = torch.linspace(-T, T, K, device=p.device, dtype=p.dtype)
    arg = p[:, :, None] * t                                         # (N, M, K)
    re, im = torch.cos(arg).mean(0), torch.sin(arg).mean(0)          # (M, K)
    phi = torch.exp(-t ** 2 / 2)
    integrand = ((re - phi) ** 2 + im ** 2) * phi
    return p.shape[0] * torch.trapezoid(integrand, t, dim=-1)       # (M,)


def sigreg(z, n_dirs=64, generator=None):
    U = torch.randn(z.shape[1], n_dirs, generator=generator).to(z)
    U = U / U.norm(dim=0, keepdim=True)
    return epps_pulley(z @ U).mean()


@torch.no_grad()
def ema_update(target, online, m):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(m).add_(po, alpha=1 - m)


def jepa_ema_loss(online, target, predictor, x_ctx, x_tgt):
    with torch.no_grad():
        tgt = target(x_tgt)
    return F.mse_loss(predictor(online(x_ctx)), tgt)
