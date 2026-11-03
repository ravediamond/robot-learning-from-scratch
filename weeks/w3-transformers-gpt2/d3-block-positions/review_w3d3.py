"""REVIEW. A 24-block policy transformer will not train: loss stays flat from step 0, deeper
is worse than shallower. Find the bug, fix, replace:
BUG:
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
        x = self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x
