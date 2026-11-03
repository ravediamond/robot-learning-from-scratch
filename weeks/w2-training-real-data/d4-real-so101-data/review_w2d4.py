"""REVIEW. Builds per-episode action arrays straight from the concatenated parquet frames.
Works on the author's machine; on the training server the policy jitters badly.
Find the bug, fix, replace:
BUG:
"""
import numpy as np
import pandas as pd


def episode_actions(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """{episode_index: (T, 6) actions in time order}."""
    out = {}
    for e, g in df.groupby("episode_index"):
        out[int(e)] = np.stack(g["action"].to_numpy())
    return out
