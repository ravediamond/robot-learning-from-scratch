"""REVIEW. SOLUTION.

BUG: w.grad is never zeroed, so each step applies the SUM of all past gradients: momentum with
no decay. w overshoots and oscillates around 3 instead of converging.
"""
import torch


def fit_gain(x: torch.Tensor, y: torch.Tensor, lr: float = 0.1, steps: int = 300) -> float:
    w = torch.zeros(1, requires_grad=True)
    for _ in range(steps):
        loss = ((x * w - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        w.grad.zero_()
    return w.item()
