"""REVIEW. Masked action loss for a chunked policy. Episodes near their end have padded
chunk steps. Training 'works'. Find the bug, fix it, replace this line:
BUG:
"""
import torch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """pred, target: (B, H, D). mask: (B, H) bool, True on real steps.

    Mean squared error over real steps only, averaged over every real (step, joint) entry.
    """
    err = (pred - target) ** 2              # (B, H, D)
    err = err * mask[..., None]             # zero the padded steps
    return err.sum() / (err.shape[0] * err.shape[1] * err.shape[2])
