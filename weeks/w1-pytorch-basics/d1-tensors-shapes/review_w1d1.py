"""REVIEW. This code was 'generated and looked fine'. It runs. It is wrong.

Find the bug, fix it, and replace this line with a one-line explanation:
BUG:
"""
import torch


def joint_stats(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-joint mean and std over a batch of states.

    batch: (N, D) with N samples and D joints. Returns mean (D,), std (D,).
    Used to normalize policy inputs.
    """
    mean = batch.mean(dim=-1)
    std = batch.std(dim=-1)
    return mean, std
