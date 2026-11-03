"""W1 D3: autograd. Do the math on paper first, then code it."""
from typing import Callable

import torch


def linear_mse_grads(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
                     y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradients of L = mean((x @ W.T + b - y)**2), computed BY HAND (no .backward()).

    x: (N, I), W: (O, I), b: (O,), y: (N, O). Returns dW (O, I), db (O,).
    """
    # TODO
    raise NotImplementedError


def finite_diff_grad(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """Central-difference gradient of a scalar function f at x. Returns same shape as x.

    Loop over elements is fine here (it is a checker, not training code).
    Do not modify x in the caller's view: work on a clone.
    """
    # TODO
    raise NotImplementedError


def fit_line(x: torch.Tensor, y: torch.Tensor, lr: float = 0.1,
             steps: int = 500) -> tuple[float, float]:
    """Fit y ≈ w * x + b with plain gradient descent and autograd (no torch.optim).

    x, y: (N,). Start from w = b = 0. Loss: mean squared error. Return (w, b) as floats.
    """
    # TODO
    raise NotImplementedError


def stop_gradient_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between pred and target where NO gradient flows into target.

    Both may require grad. After loss.backward(), target's graph must receive nothing.
    This is the stop-gradient used on the target branch of JEPA / BYOL.
    """
    # TODO
    raise NotImplementedError
