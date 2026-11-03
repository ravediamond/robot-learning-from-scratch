"""REVIEW. SOLUTION.

BUG: it resamples FRAMES, treating ~10,000 correlated frames as independent trials. The unit is the episode:
resample episodes (one outcome each). The interval was ~sqrt(frames per episode) times too narrow.
"""
import torch


def success_ci(frame_success, episode_id, B=2000, seed=0):
    eps = torch.unique(episode_id)
    outcome = torch.stack([frame_success[episode_id == e][0].float() for e in eps])     # one per episode
    g = torch.Generator().manual_seed(seed)
    n = len(outcome)
    stats = torch.stack([outcome[torch.randint(0, n, (n,), generator=g)].mean() for _ in range(B)])
    return torch.quantile(stats, 0.025).item(), torch.quantile(stats, 0.975).item()
