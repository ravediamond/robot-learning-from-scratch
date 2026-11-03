"""REVIEW. A hand-written MHA. Shapes are right, it trains, but it is clearly worse than
nn.MultiheadAttention on the same task. Find the bug, fix, replace:
BUG:
"""
import math

import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, D, h):
        super().__init__()
        self.h = h
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        split = lambda t: t.reshape(B, self.h, T, D // self.h)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        A = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D // self.h), -1)
        out = (A @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
