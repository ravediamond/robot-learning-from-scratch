"""REVIEW. Builds the training pairs for a latent world model from recorded episodes.
The predictor's loss is wonderful from the start, and it never beats copy-last.
Find the bug, fix, replace:
BUG:
"""
import torch


def make_pairs(obs: torch.Tensor, action: torch.Tensor):
    """obs (E, T+1, ...), action (E, T, 2) -> (o_t, a_t, o_next) flattened over (E, T)."""
    o_t = obs[:, :-1].flatten(0, 1)
    a_t = action.flatten(0, 1)
    o_next = obs[:, :-1].flatten(0, 1)
    return o_t, a_t, o_next
