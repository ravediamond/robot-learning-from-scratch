"""PROVIDED. Decode one camera of a LeRobot dataset to small frames, once, and cache them.
Needs the videos (download_bowls.py --videos) and PyAV (uv sync --extra video).

    uv run python extract_frames.py ~/data/so101_bowls_all_trimmed --camera front_top_camera --size 64

Writes <root>/cache/<camera>_<size>.npz with frames (N, size, size, 3) uint8 aligned with the global
'index' column of the parquet data. Handles v3 (several episodes per mp4, from/to timestamps in
meta/episodes) and v2 (one mp4 per episode).
"""
import argparse
import json
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd


def decode(path: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    frames, ts = [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        for f in c.decode(stream):
            frames.append(f.to_ndarray(format="rgb24", width=size, height=size))
            ts.append(float(f.pts * stream.time_base))
    return np.stack(frames), np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--camera", default="front_top_camera")
    ap.add_argument("--size", type=int, default=64)
    a = ap.parse_args()
    info = json.loads((a.root / "meta" / "info.json").read_text())
    fps, key = info["fps"], f"observation.images.{a.camera}"
    data = pd.concat([pd.read_parquet(f) for f in sorted((a.root / "data").rglob("*.parquet"))]).sort_values("index")
    out = np.zeros((len(data), a.size, a.size, 3), np.uint8)
    t0, n_dec = time.time(), 0
    if info["codebase_version"].startswith("v3"):
        eps = pd.concat([pd.read_parquet(f) for f in sorted((a.root / "meta" / "episodes").rglob("*.parquet"))])
        cache = {}
        for _, ep in eps.iterrows():
            ci, fi = int(ep[f"videos/{key}/chunk_index"]), int(ep[f"videos/{key}/file_index"])
            path = a.root / info["video_path"].format(video_key=key, chunk_index=ci, file_index=fi)
            if path not in cache:
                cache = {path: decode(path, a.size)}
                n_dec += len(cache[path][0])
            frames, ts = cache[path]
            rows = data[data.episode_index == ep.episode_index]
            want = float(ep[f"videos/{key}/from_timestamp"]) + rows["timestamp"].to_numpy()
            j = np.abs(ts[None, :] - want[:, None]).argmin(1)
            out[rows["index"].to_numpy() - int(data["index"].iloc[0])] = frames[j]
    else:
        for e, rows in data.groupby("episode_index"):
            path = a.root / info["video_path"].format(episode_chunk=int(e) // info.get("chunks_size", 1000),
                                                      video_key=key, episode_index=int(e))
            frames, ts = decode(path, a.size)
            n_dec += len(frames)
            j = np.abs(ts[None, :] - rows["timestamp"].to_numpy()[:, None]).argmin(1)
            out[rows["index"].to_numpy() - int(data["index"].iloc[0])] = frames[j]
    dt = time.time() - t0
    (a.root / "cache").mkdir(exist_ok=True)
    dest = a.root / "cache" / f"{a.camera}_{a.size}.npz"
    np.savez_compressed(dest, frames=out, index=data["index"].to_numpy())
    print(f"decoded {n_dec} frames in {dt:.1f}s = {n_dec / dt:.0f} frames/s  -> {dest}")


if __name__ == "__main__":
    main()
