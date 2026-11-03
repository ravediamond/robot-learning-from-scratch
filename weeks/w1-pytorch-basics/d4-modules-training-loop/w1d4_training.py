"""W1 D4: nn.Module and the training loop."""
from typing import Callable, Iterable

import torch
import torch.nn as nn


class MLP(nn.Module):
    """in_dim -> [hidden, ReLU] x n_hidden -> out_dim. Optional dropout after each ReLU.

    n_hidden=0 means a single Linear(in_dim, out_dim).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2, dropout: float = 0.0):
        super().__init__()
        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of scalar parameters. trainable_only: skip params with requires_grad=False."""
    # TODO
    raise NotImplementedError


Batch = tuple[torch.Tensor, torch.Tensor]


def train_epoch(model: nn.Module, batches: Iterable[Batch], opt: torch.optim.Optimizer,
                loss_fn: Callable) -> float:
    """One pass over batches. Returns the mean training loss (weighted by batch size)."""
    # TODO
    raise NotImplementedError


def evaluate(model: nn.Module, batches: Iterable[Batch], loss_fn: Callable) -> float:
    """Mean loss over batches (weighted by batch size). Must be deterministic and must not
    build a graph. Must leave the model in the mode it found it in."""
    # TODO
    raise NotImplementedError


def overfit_one_batch(model: nn.Module, batch: Batch, loss_fn: Callable,
                      steps: int = 500, lr: float = 1e-2) -> float:
    """Train on the same batch `steps` times with Adam. Return the final loss (float)."""
    # TODO
    raise NotImplementedError
