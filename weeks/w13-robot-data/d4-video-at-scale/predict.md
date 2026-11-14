# Predict BEFORE running `predict_run.py`

1. One 640×480 RGB camera at 30 fps, raw: GB per hour? Three cameras? →
2. 10,000 hours, 3 cameras, compressed at 1 Mbit/s per camera: TB? →
3. Keyframe every 60 frames, random frame access. Average frames decoded per sample? With a GOP of 15? →
4. Training wants 4,000 frames/s; one CPU core decodes 400 frames/s; each sample costs 30.5 decoded frames. Cores needed? →
5. 64 shards, 4 ranks × 4 workers. Shards per (rank, worker)? What happens if workers ignore their id? →

## Misses
-
