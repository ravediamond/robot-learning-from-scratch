"""REVIEW. Inference loop of a flow-matching action expert. Trained fine (low loss), but the
actions it produces at inference are wild. Find the bug, fix, replace:
BUG:
"""
import torch


@torch.no_grad()
def sample_chunk(expert, prefix, B: int, H: int, steps: int = 10, generator=None) -> torch.Tensor:
    x = torch.randn(B, H, 2, generator=generator)
    for i in range(steps):
        t = torch.full((B,), float(i))
        x = x + expert(prefix, x, t) / steps
    return x
