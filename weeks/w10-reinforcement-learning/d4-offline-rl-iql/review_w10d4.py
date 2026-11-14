"""REVIEW. IQL value loss. The trained policy is no better than behavior cloning, and V(s) sits BELOW the
dataset's average return in every state. Find the bug, fix, replace:
BUG:
"""
import torch


def value_loss(q_sa: torch.Tensor, v_s: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    diff = q_sa.detach() - v_s
    weight = torch.where(diff > 0, 1 - tau, tau)
    return (weight * diff.pow(2)).mean()
