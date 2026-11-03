"""PROVIDED driver: train YOUR PixelStatePolicy on your real bowls data (top camera, 64x64).
Needs: download_bowls.py --videos, then extract_frames.py. On a Mac: ~10-15 min.

    uv run python train_real.py ~/data/so101_bowls_all_trimmed --episodes 60

Prints val chunk MSE (raw joint units, masked) for: hold position, state-only MLP, pixel+state.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import w2d5_pixel_policy as m

sys.path.insert(0, str(Path(__file__).parent.parent / "d4-real-so101-data"))
import w2d4_lerobot as io  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("--episodes", type=int, default=60)
ap.add_argument("--H", type=int, default=10)
ap.add_argument("--epochs", type=int, default=15)
a = ap.parse_args()
frames = np.load(a.root / "cache" / "front_top_camera_64.npz")["frames"]
df = io.load_frames(a.root)
eps = io.to_episodes(df)
rng = np.random.default_rng(0)
chosen = rng.permutation(len(eps))[: a.episodes]
n_val = max(2, len(chosen) // 5)
starts = np.concatenate([[0], np.cumsum([len(e["obs"]) for e in eps])])


def build(ids):
    img, st, act, mask = [], [], [], []
    for e in ids:
        ep, T = eps[e], len(eps[e]["obs"])
        for t in range(0, T, 2):
            steps = np.arange(t, t + a.H)
            img.append(frames[starts[e] + t])
            st.append(ep["obs"][t])
            act.append(ep["action"][np.minimum(steps, T - 1)])
            mask.append(steps < T)
    return {"img": torch.tensor(np.stack(img)).permute(0, 3, 1, 2).float() / 255, "state": torch.tensor(np.stack(st)),
            "actions": torch.tensor(np.stack(act)), "mask": torch.tensor(np.stack(mask))}


train, val = build(chosen[n_val:]), build(chosen[:n_val])
print(f"train {len(train['state'])} samples, val {len(val['state'])} samples ({n_val} episodes)")
hold = m.masked_chunk_mse(val["state"][:, None, :].expand_as(val["actions"]), val["actions"], val["mask"]).item()
state_only = m.train_policy(train, use_image=False, epochs=a.epochs)
pixel = m.train_policy(train, use_image=True, epochs=a.epochs)
print(f"val chunk MSE  hold: {hold:.2f}   state-only: {m.evaluate(state_only, val):.2f}   pixel+state: {m.evaluate(pixel, val):.2f}")
