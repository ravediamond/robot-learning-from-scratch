"""REVIEW. One CEM iteration. The planner never improves over random shooting, whatever the number
of iterations. Find the bug, fix, replace:
BUG:
"""
import torch


def cem_iteration(samples: torch.Tensor, cost: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """samples (N, H, A) action sequences, cost (N,). Returns the new (mu, std), each (H, A)."""
    elite_idx = torch.topk(cost, K, largest=False).indices
    elites = samples[elite_idx]
    return samples.mean(0), samples.std(0) + 1e-3
