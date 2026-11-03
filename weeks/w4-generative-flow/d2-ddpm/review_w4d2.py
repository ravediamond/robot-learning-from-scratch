"""REVIEW. SOLUTION.

BUG: the coefficients are swapped: early steps are almost pure noise and late steps almost clean data,
the opposite of what the sampler (which starts from noise at the last step) assumes.
"""
import torch


def noise_chunks(x0, t, noise, abar):
    a = abar[t][:, None, None]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise
