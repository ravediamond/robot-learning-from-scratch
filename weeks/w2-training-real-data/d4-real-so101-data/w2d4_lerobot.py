"""W2 D4: read a LeRobot dataset with pandas, no lerobot import."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_info(root: str | Path) -> dict:
    """Parse meta/info.json."""
    # TODO
    raise NotImplementedError


def load_frames(root: str | Path) -> pd.DataFrame:
    """Read EVERY parquet file under <root>/data (any depth), concatenate, sort by the global
    'index' column, reset the DataFrame index. Works for v3 (file-XXX) and v2 (episode_XXXXXX)."""
    # TODO
    raise NotImplementedError


def to_episodes(df: pd.DataFrame) -> list[dict]:
    """One dict per episode, ordered by episode_index, rows in frame_index order:
    {"episode_index": int, "task_index": int,
     "obs": (T, 6) float32 np array from observation.state, "action": (T, 6) float32, "timestamp": (T,) float}"""
    # TODO
    raise NotImplementedError


def check_episode(ep: dict, fps: float, idle_thresh: float = 0.5) -> dict:
    """{"gaps": number of timestamp steps > 1.5/fps,
        "nonfinite": True if any nan/inf in obs or action,
        "idle_start": idle frames after frame 0: find the first t>0 with
                      max |action[t] - action[0]| > idle_thresh; idle_start = t - 1 (0 if it moves at t=1)}"""
    # TODO
    raise NotImplementedError


def action_state_lag(ep: dict, max_lag: int = 10) -> int:
    """Integer k in [0, max_lag] minimizing mean |obs[t + k] - action[t]| over valid t
    (all 6 dims). The number of frames the follower trails the command."""
    # TODO
    raise NotImplementedError
