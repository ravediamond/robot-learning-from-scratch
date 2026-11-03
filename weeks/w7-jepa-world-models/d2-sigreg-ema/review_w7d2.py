"""REVIEW. EMA update of the target encoder in an I-JEPA-style training loop, m = 0.996.
The target is supposed to move slowly; in practice it is a near copy of the online encoder
after every step, and training collapses. Find the bug, fix, replace:
BUG:
"""
import torch


@torch.no_grad()
def ema(target, online, m: float = 0.996):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.copy_(m * po + (1 - m) * pt)
