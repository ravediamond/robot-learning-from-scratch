# W2 D4: Your real SO-101 data, by hand

Book: 10b (the data pipeline), 10a.5 (two clocks). Data: `ravediamond/so101_bowls_all_trimmed`
(182 episodes, 61,462 frames at 30 fps, 2 cameras, idle starts trimmed: register row 2026-10-01).

## The one idea
A LeRobot dataset is just parquet tables + mp4 videos + a `meta/info.json`. No magic.
Read it yourself once, and you will never again wonder what the loader did.

```python
pd.read_parquet(path)                         # a DataFrame; array columns hold np arrays (object dtype)
np.stack(df["action"].to_numpy())             # (N, 6) float32
df.groupby("episode_index")                   # one group per episode
sorted(Path(root, "data").rglob("*.parquet")) # v3: file-000, file-001...; v2: episode_000000...
```
Every frame row has `episode_index`, `frame_index`, `timestamp`, `index` (global), `task_index`.

## What real data does that toy data didn't
- **Gaps**: dropped frames show up as `timestamp` steps larger than 1/fps.
- **Idle starts**: the operator waits before moving. A policy trained on them learns to freeze
  (your decision D26: trim them).
- **Lag**: `action` is the leader-arm command; `observation.state` is where the follower is.
  The follower trails the command by a few frames. That lag is why "action = state" (hold
  position) is not a zero-error baseline, and why chunked policies help.

## Today's build (`w2d4_lerobot.py`)
`read_info`, `load_frames` (all parquet files, sorted by global `index`), `to_episodes`,
`check_episode` (gaps, non-finite, idle start), `action_state_lag`.

Tests run on a fake dataset in the same layout (`fake_lerobot.py`). Then, on your Mac:
```bash
uv run python download_bowls.py
uv run python inspect_real.py
```
and compare with the register numbers. Write the output in `PROGRESS.md`.
