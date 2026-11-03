"""W2 D3: SOLUTION."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_unfold(x, weight, bias, stride=1, padding=0):
    B, C, H, W = x.shape
    Co, _, k, _ = weight.shape
    cols = F.unfold(x, k, padding=padding, stride=stride)          # (B, C*k*k, L)
    out = weight.view(Co, -1) @ cols                               # (B, Co, L)
    if bias is not None:
        out = out + bias[None, :, None]
    Ho = (H + 2 * padding - k) // stride + 1
    Wo = (W + 2 * padding - k) // stride + 1
    return out.view(B, Co, Ho, Wo)


def receptive_field(layers):
    r, j = 1, 1
    for k, s in layers:
        r += (k - 1) * j
        j *= s
    return r


class SpatialSoftmax(nn.Module):
    def forward(self, feat):
        B, C, H, W = feat.shape
        p = torch.softmax(feat.flatten(2), dim=-1).view(B, C, H, W)   # (B, C, H, W)
        ys = torch.linspace(-1, 1, H, device=feat.device)
        xs = torch.linspace(-1, 1, W, device=feat.device)
        ex = (p.sum(2) * xs).sum(-1)                                   # (B, C): marginal over H
        ey = (p.sum(3) * ys).sum(-1)                                   # (B, C)
        return torch.stack([ex, ey], dim=-1).flatten(1)                # (B, 2C)


class CNNEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.GroupNorm(8, 32), nn.ReLU(),
        )
        self.pool = SpatialSoftmax()
        self.fc = nn.Linear(64, out_dim)

    def forward(self, img):
        return self.fc(self.pool(self.net(img)))


def random_shift(imgs, pad, generator=None):
    if pad == 0:
        return imgs
    B, C, H, W = imgs.shape
    padded = F.pad(imgs, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), generator=generator)
    oy = torch.randint(0, 2 * pad + 1, (B,), generator=generator)
    rows = (oy[:, None] + torch.arange(H)[None])                      # (B, H)
    cols = (ox[:, None] + torch.arange(W)[None])                      # (B, W)
    b = torch.arange(B)[:, None, None]
    out = padded.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]   # (B, H, W, C)
    return out.permute(0, 3, 1, 2).contiguous()
