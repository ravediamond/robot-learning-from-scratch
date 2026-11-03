"""REVIEW. Rotation about y for the SO-101's pitch joints. FK runs, poses look plausible, but when
the shoulder 'lifts' the simulated arm goes DOWN into the table. Find the bug, fix, replace:
BUG:
"""
import torch


def rot_y(t: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(t), torch.sin(t)
    one, zero = torch.ones_like(t), torch.zeros_like(t)
    return torch.stack([torch.stack([c, zero, -s]), torch.stack([zero, one, zero]), torch.stack([s, zero, c])])
