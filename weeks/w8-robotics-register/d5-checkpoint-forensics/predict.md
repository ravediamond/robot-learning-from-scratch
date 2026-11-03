# Predict BEFORE running `predict_run.py` (after your tests pass)

1. A checkpoint saved from a DDP-wrapped model, loaded into a plain model with strict=False: how many tensors are loaded? →
2. Same with strict=True →
3. The policy built from that silent load: its outputs on a fixed input vs the trained policy: same? →
4. How long would it take to notice in a training run that "fine-tunes" from this checkpoint? What would you see? →

## Misses
-
