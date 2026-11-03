"""REVIEW. A policy with dropout. Validation loss is used to pick the best checkpoint.
Two runs on the same checkpoint give different validation numbers, and the 'best' checkpoint
changes between reruns. Find the bug(s), fix, replace this line:
BUG:
"""
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(6, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 6)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"l{i}", layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def val_loss(policy: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    pred = policy(x)
    return nn.functional.mse_loss(pred, y).item()
