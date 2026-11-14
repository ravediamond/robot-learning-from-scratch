"""REVIEW. Advantage labelling for a two-task dataset (folding: returns around -5; espresso: around -50).
After training, asking for I = 1 on the espresso task changes nothing. Find the bug, fix, replace:
BUG:
"""
import torch


def label(adv: torch.Tensor, task_id: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    threshold = torch.quantile(adv, q)
    return adv > threshold
