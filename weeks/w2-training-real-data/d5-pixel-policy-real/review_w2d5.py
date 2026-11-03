"""REVIEW. Evaluation of an image policy. The val number is better than the train number,
every time, by a lot. The author is happy. Find the bug, fix, replace:
BUG:
"""
import torch

from cnn_ref import random_shift


def evaluate_policy(policy, data, aug_pad: int = 4) -> float:
    """Masked chunk MSE in raw units on held-out data."""
    policy.eval()
    with torch.no_grad():
        img = random_shift(data["img"], aug_pad)
        pred = policy(img, data["state"])
        err = ((pred - data["actions"]) ** 2).mean(-1)
        return (err * data["mask"]).sum().item() / data["mask"].sum().item()
