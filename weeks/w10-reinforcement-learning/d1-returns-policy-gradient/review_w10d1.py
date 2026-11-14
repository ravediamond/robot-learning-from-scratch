"""REVIEW. Return computation from the PPO branch. The rollout buffer holds several episodes back to back
per env (auto-reset). Value-function targets look too large at the END of each episode. Find the bug, fix, replace:
BUG:
"""
import torch


def returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    """rewards, dones (T, N). dones[t] = the episode ended at step t."""
    G = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * running
        G[t] = running
    return G
