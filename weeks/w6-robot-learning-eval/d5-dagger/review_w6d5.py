"""REVIEW. SOLUTION.

BUG: the visited states are labelled with the POLICY's own actions, so the "new data" only repeats what
the policy already does (self-training); nothing teaches recovery. Label them with the expert.
"""
import torch


def dagger_step(policy, expert, obs, act, visited):
    with torch.no_grad():
        labels = expert(visited)
    return torch.cat([obs, visited]), torch.cat([act, labels])
