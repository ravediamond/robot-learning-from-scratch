"""REVIEW. SOLUTION.

BUG: the window only stops at the end of the whole tensor, not at the end of t's episode, so the last
H-1 frames of every episode get targets from the START of the next episode (a teleport in the action).
"""
import torch


def window_actions(actions, episode_index, t: int, H: int):
    last = int((episode_index == episode_index[t]).nonzero().max())     # last frame of t's episode
    steps = torch.arange(t, t + H)
    mask = steps <= last
    return actions[steps.clamp(max=last)], mask
