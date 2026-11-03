"""REVIEW. SOLUTION.

BUG: the time passed to the network is the step index (0..9) instead of t = i / steps in [0, 1): the
network is queried at times it never saw in training. Use t = i / steps.
"""
import torch


@torch.no_grad()
def sample_chunk(expert, prefix, B, H, steps=10, generator=None):
    x = torch.randn(B, H, 2, generator=generator)
    for i in range(steps):
        x = x + expert(prefix, x, torch.full((B,), i / steps)) / steps
    return x
