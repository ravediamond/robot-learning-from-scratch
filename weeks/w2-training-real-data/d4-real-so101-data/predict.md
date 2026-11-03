# Predict BEFORE running `predict_run.py`

1. A parquet column holding a 6-float array per row: `df["action"].dtype` →
2. `np.stack(df["action"].to_numpy()).shape` for 100 rows →
3. `sorted(["file-10.parquet", "file-9.parquet", "file-100.parquet"])` →
4. `df.groupby("episode_index")` on rows that are NOT sorted by frame: are rows inside a group in time order? →
5. Follower servo lags the command by 2 frames at 30 fps: lag in ms →
6. On your real data, guess the median action → state lag in frames →   (check tonight with inspect_real.py)

## Misses
-
