"""REVIEW. SOLUTION.

BUG: the expectile weights are swapped. With tau = 0.8 the LARGE weight must sit on positive errors
(Q above V), pulling V up towards the best dataset actions. Swapped, V becomes the 0.2-expectile:
a pessimistic value, and the advantages no longer single out the good actions.
"""
import torch


def value_loss(q_sa: torch.Tensor, v_s: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    diff = q_sa.detach() - v_s
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()
