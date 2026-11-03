"""PROVIDED. Writes a small fake dataset in the LeRobot v3.0 on-disk layout, with the same
columns as your SO-101 datasets, so the tests run offline. Do not edit.

Layout (same as ravediamond/so101_bowls_all_trimmed if it is v3.0; v2.x has one parquet
per episode under data/chunk-000/episode_XXXXXX.parquet: your loader should handle both
by reading every parquet under data/):
    meta/info.json                      fps, features, totals
    meta/tasks.parquet                  task string -> task_index
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet     frames: action, observation.state, timestamp,
    data/chunk-000/file-001.parquet       frame_index, episode_index, index, task_index
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["put the blue bowl in the pink bowl", "put the pink bowl in the blue bowl"]


def make_fake_dataset(root: str | Path, n_episodes: int = 6, fps: int = 30, lag: int = 2, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    rows, eps, index = [], [], 0
    for e in range(n_episodes):
        T = int(rng.integers(40, 80))
        idle = 8 if e % 3 == 0 else 0                           # some episodes start idle
        goal = rng.uniform(-60, 60, 6)
        act = np.zeros((T, 6), np.float32)
        a = rng.uniform(-30, 30, 6)
        for t in range(T):
            if t >= idle:
                a = a + 0.15 * (goal - a) + rng.normal(0, 0.3, 6)
            act[t] = a
        state = np.concatenate([np.repeat(act[:1], lag, 0), act[:-lag]], 0)   # servo follows with a lag
        state = state + rng.normal(0, 0.05, state.shape)
        ts = np.arange(T, dtype=np.float32) / fps
        if e == 1:
            ts[20:] += 5 / fps                                  # a dropped-frames gap
        for t in range(T):
            rows.append({"action": act[t].astype(np.float32), "observation.state": state[t].astype(np.float32),
                         "timestamp": np.float32(ts[t]), "frame_index": t, "episode_index": e,
                         "index": index + t, "task_index": e % 2})
        eps.append({"episode_index": e, "tasks": [TASKS[e % 2]], "length": T,
                    "dataset_from_index": index, "dataset_to_index": index + T})
        index += T
    df = pd.DataFrame(rows)
    half = len(df) // 2
    df.iloc[:half].to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    df.iloc[half:].to_parquet(root / "data" / "chunk-000" / "file-001.parquet")
    pd.DataFrame(eps).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pd.DataFrame({"task_index": [0, 1]}, index=pd.Index(TASKS, name="task")).to_parquet(root / "meta" / "tasks.parquet")
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v3.0", "fps": fps, "total_episodes": n_episodes, "total_frames": index,
        "robot_type": "so_follower",
        "features": {"action": {"dtype": "float32", "shape": [6]}, "observation.state": {"dtype": "float32", "shape": [6]}}}))
    return root
