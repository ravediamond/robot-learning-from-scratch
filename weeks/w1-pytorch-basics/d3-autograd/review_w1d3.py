"""REVIEW. A scalar gain is fitted to calibration data (y = gain * x).
It runs without error and returns a number. Find the bug, fix it, replace this line:
BUG:
"""
import torch


def fit_gain(x: torch.Tensor, y: torch.Tensor, lr: float = 0.1, steps: int = 300) -> float:
    w = torch.zeros(1, requires_grad=True)
    for _ in range(steps):
        loss = ((x * w - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
    return w.item()
