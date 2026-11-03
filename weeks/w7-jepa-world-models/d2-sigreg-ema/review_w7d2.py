"""REVIEW. SOLUTION.

BUG: the weights are swapped: m * online + (1 - m) * target puts 99.6% of the ONLINE weights into the
target every step, so the target is a copy of the online network (no slow target, collapse returns).
"""
import torch


@torch.no_grad()
def ema(target, online, m=0.996):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.copy_(m * pt + (1 - m) * po)
