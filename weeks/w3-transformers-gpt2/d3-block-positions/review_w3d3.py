"""REVIEW. SOLUTION.

BUG: the attention sublayer replaces x instead of adding to it (missing residual). Every block discards
its input; with 24 blocks the input signal and gradients must pass through 24 attention layers.
"""
import torch.nn as nn

from mha_ref import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.ln_1, self.ln_2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn = MultiHeadAttention(D, h)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        return x + self.mlp(self.ln_2(x))
