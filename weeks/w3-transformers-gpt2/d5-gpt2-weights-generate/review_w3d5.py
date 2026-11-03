"""REVIEW. SOLUTION.

BUG: temperature divides the PROBABILITIES, and the renormalization cancels it exactly: no effect.
Temperature must divide the LOGITS before the softmax.
"""
import torch


def sample_next(logits, temperature, generator=None):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)
