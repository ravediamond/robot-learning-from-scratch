"""REVIEW. The forward (noising) process of a diffusion policy. Training loss goes down, but
sampled action chunks are pure noise. Find the bug, fix, replace:
BUG:
"""
import torch


def noise_chunks(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, abar: torch.Tensor) -> torch.Tensor:
    """x0 (N, H, D) clean chunks, t (N,) step indices -> x_t."""
    a = abar[t][:, None, None]
    return (1 - a).sqrt() * x0 + a.sqrt() * noise
