"""W2 D4: SOLUTION."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_info(root):
    return json.loads((Path(root) / "meta" / "info.json").read_text())


def load_frames(root):
    files = sorted((Path(root) / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)


def to_episodes(df):
    eps = []
    for e, g in df.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        eps.append({"episode_index": int(e), "task_index": int(g["task_index"].iloc[0]),
                    "obs": np.stack(g["observation.state"].to_numpy()).astype(np.float32),
                    "action": np.stack(g["action"].to_numpy()).astype(np.float32),
                    "timestamp": g["timestamp"].to_numpy().astype(float)})
    return eps


def check_episode(ep, fps, idle_thresh=0.5):
    gaps = int((np.diff(ep["timestamp"]) > 1.5 / fps).sum())
    nonfinite = not (np.isfinite(ep["obs"]).all() and np.isfinite(ep["action"]).all())
    moved = np.abs(ep["action"] - ep["action"][0]).max(axis=1) > idle_thresh   # (T,)
    first = int(np.argmax(moved)) if moved.any() else len(moved)
    return {"gaps": gaps, "nonfinite": nonfinite, "idle_start": max(first - 1, 0)}


def action_state_lag(ep, max_lag=10):
    a, s = ep["action"], ep["obs"]
    errs = [np.abs(s[k:] - a[: len(a) - k]).mean() for k in range(max_lag + 1)]
    return int(np.argmin(errs))
