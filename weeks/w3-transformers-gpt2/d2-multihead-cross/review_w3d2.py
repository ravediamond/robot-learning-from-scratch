"""REVIEW. SOLUTION.

BUG: reshape(B, h, T, Dh) splits the memory as if heads came before time: each "head" gets a block of
consecutive time steps' features. Must split the feature dim: view(B, T, h, Dh).transpose(1, 2).
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
        split = lambda t: t.view(B, T, self.h, D // self.h).transpose(1, 2)
        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        A = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(D // self.h), -1)
        return self.out_proj((A @ v).transpose(1, 2).reshape(B, T, D))
