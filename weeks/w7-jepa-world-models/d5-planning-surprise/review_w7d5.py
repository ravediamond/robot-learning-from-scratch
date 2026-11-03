"""REVIEW. SOLUTION.

BUG: the elites are selected and then ignored: mu and std are refitted on ALL samples, so the distribution
never moves toward low cost (it just stays where it was: random shooting). Refit on the elites.
"""
import torch


def cem_iteration(samples, cost, K):
    elites = samples[torch.topk(cost, K, largest=False).indices]
    return elites.mean(0), elites.std(0) + 1e-3
