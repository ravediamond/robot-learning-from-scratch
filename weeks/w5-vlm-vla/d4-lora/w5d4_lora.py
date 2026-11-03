"""W5 D4: SOLUTION."""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(r))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    @torch.no_grad()
    def merged(self):
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def apply_lora(model, names, r=8, alpha=16.0):
    for p in model.parameters():
        p.requires_grad_(False)
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha))
    return model


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
