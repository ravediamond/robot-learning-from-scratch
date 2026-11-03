"""PROVIDED driver. After your tests pass AND download_bowls.py ran, inspect the real data
with YOUR functions:   uv run python inspect_real.py [path]
Register says: 182 episodes, 61,462 frames, 30 fps. Do your numbers agree?"""
import sys
from pathlib import Path

import numpy as np

import w2d4_lerobot as m

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data" / "so101_bowls_all_trimmed"
info = m.read_info(root)
eps = m.to_episodes(m.load_frames(root))
print(f"codebase {info.get('codebase_version')}  fps {info['fps']}  episodes {len(eps)}  frames {sum(len(e['obs']) for e in eps)}")
lens = np.array([len(e["obs"]) for e in eps])
print(f"length min/median/max: {lens.min()} / {int(np.median(lens))} / {lens.max()}")
issues = [m.check_episode(e, info["fps"]) for e in eps]
print(f"episodes with timestamp gaps: {sum(i['gaps'] > 0 for i in issues)}   non-finite: {sum(i['nonfinite'] for i in issues)}")
print(f"idle start frames: mean {np.mean([i['idle_start'] for i in issues]):.1f}  max {max(i['idle_start'] for i in issues)}")
lags = [m.action_state_lag(e) for e in eps[:30]]
print(f"action -> state lag (frames), first 30 episodes: {sorted(set(lags))}  median {int(np.median(lags))}")
