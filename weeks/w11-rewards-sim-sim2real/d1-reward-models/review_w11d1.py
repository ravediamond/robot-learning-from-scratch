"""REVIEW. SOLUTION.

BUG: a and b are drawn from the whole dataset, so most pairs mix two episodes. Their order in time says
nothing about progress (different starts and goals); the model can only learn "later = better". Draw the
episode first, then two different times inside it.
"""
import torch


def sample_pairs(episode: torch.Tensor, t: torch.Tensor, n_pairs: int, generator=None):
    n_ep = int(episode.max().item()) + 1
    T = len(episode) // n_ep
    e = torch.randint(0, n_ep, (n_pairs,), generator=generator)
    ta = torch.randint(0, T, (n_pairs,), generator=generator)
    tb = (ta + torch.randint(1, T, (n_pairs,), generator=generator)) % T
    idx_a, idx_b = ta * n_ep + e, tb * n_ep + e
    return idx_a, idx_b, (t[idx_b] > t[idx_a]).float()
