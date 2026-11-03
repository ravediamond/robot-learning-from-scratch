"""REVIEW. Builds a flow-matching training batch for an action head. The sampler (not shown)
starts from Gaussian noise at t = 0 and integrates to t = 1. The policy outputs garbage.
Find the bug, fix, replace:
BUG:
"""
import torch


def make_fm_batch(actions: torch.Tensor, generator: torch.Generator):
    """actions (N, H, D) clean chunks -> (x_t, t, target_velocity)."""
    noise = torch.randn(actions.shape, generator=generator)
    t = torch.rand(actions.shape[0], generator=generator)
    tt = t[:, None, None]
    x_t = tt * noise + (1 - tt) * actions
    target = actions - noise
    return x_t, t, target
