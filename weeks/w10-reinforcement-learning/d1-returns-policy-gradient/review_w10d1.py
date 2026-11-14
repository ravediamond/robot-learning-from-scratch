"""REVIEW. SOLUTION.

BUG: `dones` is never used, so the return of the last steps of one episode includes the rewards of the NEXT
episode in the buffer. Cut the recursion at episode ends: running = r_t + gamma * (1 - done_t) * running.
"""
import torch


def returns(rewards: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
    G = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[t] + gamma * (~dones[t]).float() * running
        G[t] = running
    return G
