"""REVIEW. Train/validation split of the team's policy-training script. Validation success-prediction is
excellent and real-robot results are much worse; the dataset has 40 sessions with different lighting.
Find the bug, fix, replace:
BUG:
"""
import random


def split(episode_sessions: list[str], val_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    """episode_sessions[i] = session of episode i. Returns (train ids, val ids)."""
    ids = list(range(len(episode_sessions)))
    random.Random(seed).shuffle(ids)
    k = int(val_frac * len(ids))
    return sorted(ids[k:]), sorted(ids[:k])
