"""REVIEW. SOLUTION.

BUG: A and B are both initialised to zero. dL/dB = scale * grad_out^T (x A^T) = 0 because A = 0, and
dL/dA = scale * (grad_out B)^T x = 0 because B = 0: no gradient ever. Initialise A random, B zero.
"""
import math

import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T
