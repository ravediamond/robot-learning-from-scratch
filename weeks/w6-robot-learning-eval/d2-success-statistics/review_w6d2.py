"""REVIEW. Confidence interval for a policy's success rate from the logged rollouts table
(one row per FRAME, with the episode's outcome repeated on each row). The interval is
impressively tight: "93% ± 0.6% with only 40 episodes". Find the bug, fix, replace:
BUG:
"""
import torch


def success_ci(frame_success: torch.Tensor, episode_id: torch.Tensor, B: int = 2000, seed: int = 0):
    """frame_success (F,) 0/1, episode_id (F,) -> (lo, hi) 95% CI of the per-episode success rate."""
    g = torch.Generator().manual_seed(seed)
    F = len(frame_success)
    stats = torch.stack([frame_success[torch.randint(0, F, (F,), generator=g)].float().mean() for _ in range(B)])
    return torch.quantile(stats, 0.025).item(), torch.quantile(stats, 0.975).item()
