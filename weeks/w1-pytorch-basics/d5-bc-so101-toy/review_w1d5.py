"""REVIEW. Splitting a LeRobot-style dataset into train/val. `episode_index` gives, for every
frame, which episode it belongs to (as in LeRobotDataset). The resulting val loss was
excellent. Find the bug, fix it, replace this line:
BUG:
"""
import torch


def make_val_mask(episode_index: torch.Tensor, val_frac: float = 0.2, seed: int = 0) -> torch.Tensor:
    """episode_index: (N,) long. Returns a (N,) bool mask, True for validation frames.

    About val_frac of the data goes to validation.
    """
    g = torch.Generator().manual_seed(seed)
    N = episode_index.shape[0]
    perm = torch.randperm(N, generator=g)
    mask = torch.zeros(N, dtype=torch.bool)
    mask[perm[: int(val_frac * N)]] = True
    return mask
