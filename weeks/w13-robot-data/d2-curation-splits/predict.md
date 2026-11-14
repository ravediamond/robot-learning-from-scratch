# Predict BEFORE running `predict_run.py`

1. 20 sessions of 10 episodes, each session with its own lighting. Random 80/20 episode split: how many validation episodes share a session with training episodes? →
2. Group split by session: same question →
3. Two trajectories of different lengths (240 and 300 frames) of the same motion at different speeds. After resampling both to 50 points, is their distance small or large? →
4. You add one episode and remove another. Does the dataset fingerprint change? And if you only reorder episodes? →

## Misses
-
