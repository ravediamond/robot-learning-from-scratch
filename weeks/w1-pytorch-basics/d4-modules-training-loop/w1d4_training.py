"""W1 D4: SOLUTION."""
from typing import Callable, Iterable

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (..., in_dim) -> (..., out_dim)


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


Batch = tuple[torch.Tensor, torch.Tensor]


def train_epoch(model, batches: Iterable[Batch], opt, loss_fn: Callable) -> float:
    model.train()
    total, n = 0.0, 0
    for xb, yb in batches:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / n


def evaluate(model, batches: Iterable[Batch], loss_fn: Callable) -> float:
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in batches:
            total += loss_fn(model(xb), yb).item() * len(xb)
            n += len(xb)
    model.train(was_training)
    return total / n


def overfit_one_batch(model, batch: Batch, loss_fn: Callable, steps: int = 500, lr: float = 1e-2) -> float:
    xb, yb = batch
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()
