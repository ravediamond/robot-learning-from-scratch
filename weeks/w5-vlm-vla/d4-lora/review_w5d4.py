"""REVIEW. A LoRA adapter for fine-tuning a VLA's language model on SO-101 instructions.
The training loss is exactly flat from step 0; the adapter weights never move.
Find the bug, fix, replace:
BUG:
"""
import torch
import torch.nn as nn


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T
