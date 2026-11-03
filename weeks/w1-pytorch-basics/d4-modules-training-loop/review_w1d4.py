"""REVIEW. SOLUTION.

BUG: val_loss runs with dropout active (no model.eval()) and builds a graph (no no_grad), so
validation is random and checkpoint selection is noise. Fix: eval + no_grad, restore the mode.
(The add_module loop is clumsy but correct: nn.ModuleList would be the clean version.)
"""
import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(6, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 6)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def val_loss(policy: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        loss = nn.functional.mse_loss(policy(x), y).item()
    policy.train(was_training)
    return loss
