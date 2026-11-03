"""REVIEW. "DAgger" implementation for an arm policy. Three iterations, the dataset tripled, and
the success rate under disturbances did not move at all. Find the bug, fix, replace:
BUG:
"""
import torch


def dagger_step(policy, expert, obs, act, visited):
    """Aggregate the states visited by `policy` into the dataset (obs, act)."""
    with torch.no_grad():
        labels = policy(visited)
    return torch.cat([obs, visited]), torch.cat([act, labels])
