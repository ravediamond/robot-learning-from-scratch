"""REVIEW. SOLUTION.

BUG: episodes are split at random, so almost every validation episode has siblings from the same session in
training: same lighting, objects, operator habits. Validation then measures memorisation of the session.
Split by session (a whole session goes to one side), so validation measures generalisation to new sessions.
"""
import random


def split(episode_sessions: list[str], val_frac: float = 0.2, seed: int = 0) -> tuple[list[int], list[int]]:
    sessions = sorted(set(episode_sessions))
    random.Random(seed).shuffle(sessions)
    target, val_sessions, n = val_frac * len(episode_sessions), set(), 0
    for s in sessions:
        if n >= target:
            break
        val_sessions.add(s)
        n += episode_sessions.count(s)
    val = [i for i, s in enumerate(episode_sessions) if s in val_sessions]
    train = [i for i, s in enumerate(episode_sessions) if s not in val_sessions]
    return train, val
