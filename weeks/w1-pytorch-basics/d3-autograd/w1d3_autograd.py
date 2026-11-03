"""W1 D3: SOLUTION."""
from typing import Callable

import torch


def linear_mse_grads(x, W, b, y):
    N, O = y.shape
    pred = x @ W.T + b                     # (N, O)
    g = 2.0 * (pred - y) / (N * O)         # dL/dpred, (N, O)
    dW = g.T @ x                           # (O, N) @ (N, I) -> (O, I)
    db = g.sum(dim=0)                      # (O,)
    return dW, db


def finite_diff_grad(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    x = x.clone()
    grad = torch.zeros_like(x)
    flat_x, flat_g = x.view(-1), grad.view(-1)
    for i in range(flat_x.numel()):
        old = flat_x[i].item()
        flat_x[i] = old + eps
        f_plus = f(x).item()
        flat_x[i] = old - eps
        f_minus = f(x).item()
        flat_x[i] = old
        flat_g[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def fit_line(x, y, lr: float = 0.1, steps: int = 500):
    w = torch.zeros((), requires_grad=True)
    b = torch.zeros((), requires_grad=True)
    for _ in range(steps):
        loss = ((w * x + b - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    return w.item(), b.item()


def stop_gradient_mse(pred, target):
    return ((pred - target.detach()) ** 2).mean()
