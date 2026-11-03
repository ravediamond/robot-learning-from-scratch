"""REVIEW. SOLUTION.

BUG: frames are split at random, so almost every val frame has its near-duplicate neighbors in
train (same episode): val loss measures memorization, not generalization to new episodes.
Fix: choose val EPISODES, then mark all their frames.
"""
import torch


def make_val_mask(episode_index: torch.Tensor, val_frac: float = 0.2, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    episodes = torch.unique(episode_index)
    perm = episodes[torch.randperm(len(episodes), generator=g)]
    val_eps = perm[: max(1, round(val_frac * len(episodes)))]
    return torch.isin(episode_index, val_eps)
