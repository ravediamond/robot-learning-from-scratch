"""REVIEW. Pair sampler for the progress model. The learned reward rises with elapsed time even when the arm
does nothing, and RL learns to wait. Find the bug, fix, replace:
BUG:
"""
import torch


def sample_pairs(episode: torch.Tensor, t: torch.Tensor, n_pairs: int, generator=None):
    """Rows are time-major: row = t * n_episodes + e. Returns (idx_a, idx_b, label: b later than a)."""
    n = len(episode)
    idx_a = torch.randint(0, n, (n_pairs,), generator=generator)
    idx_b = torch.randint(0, n, (n_pairs,), generator=generator)
    keep = t[idx_a] != t[idx_b]
    idx_a, idx_b = idx_a[keep], idx_b[keep]
    return idx_a, idx_b, (t[idx_b] > t[idx_a]).float()
