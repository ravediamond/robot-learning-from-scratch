"""REVIEW. SOLUTION.

BUG: the denominator counts padded entries (B*H*D), so episodes near their end get a smaller
loss: the policy learns the last steps of each episode (grasp, place) less than the rest.
"""
import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    err = (pred - target) ** 2                    # (B, H, D)
    m = mask[..., None].to(err.dtype)             # (B, H, 1)
    err = err * m
    n_valid = m.sum() * err.shape[-1]             # real steps × joints
    return err.sum() / n_valid.clamp(min=1.0)
