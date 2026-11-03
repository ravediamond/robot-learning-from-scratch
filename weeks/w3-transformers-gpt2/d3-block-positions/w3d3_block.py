"""W3 D3: SOLUTION."""
import math

import torch
import torch.nn as nn

from mha_ref import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = MultiHeadAttention(D, h)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask=None, context=None):
        x = x + self.attn(self.ln_1(x), context=context, mask=mask)
        return x + self.mlp(self.ln_2(x))


def sinusoidal_positions(T, D):
    pos = torch.arange(T, dtype=torch.float32)[:, None]                   # (T, 1)
    div = 10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D)     # (D/2,)
    pe = torch.zeros(T, D)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe


def apply_rope(x, positions, base=10000.0):
    d = x.shape[-1]
    theta = base ** (-torch.arange(0, d, 2, dtype=torch.float32, device=x.device) / d)    # (d/2,)
    ang = positions.float().to(x.device)[:, None] * theta                                 # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)


def patchify(img, p):
    B, C, H, W = img.shape
    x = img.view(B, C, H // p, p, W // p, p)                 # (B, C, Hp, p, Wp, p)
    x = x.permute(0, 2, 4, 1, 3, 5)                          # (B, Hp, Wp, C, p, p)
    return x.reshape(B, (H // p) * (W // p), C * p * p)
