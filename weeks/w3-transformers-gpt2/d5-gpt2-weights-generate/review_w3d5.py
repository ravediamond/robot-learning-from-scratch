"""REVIEW. Sampling with temperature. Low temperatures were supposed to make text more
conservative; they make no difference at all. Find the bug, fix, replace:
BUG:
"""
import torch


def sample_next(logits: torch.Tensor, temperature: float, generator=None) -> torch.Tensor:
    """logits (B, V) for the last position -> (B, 1) sampled token ids."""
    probs = torch.softmax(logits, dim=-1)
    probs = probs / temperature
    probs = probs / probs.sum(-1, keepdim=True)
    return torch.multinomial(probs, 1, generator=generator)
