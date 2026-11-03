# Predict BEFORE running `predict_run.py` (after your tests pass)

1. SIGReg of N(0, I) samples (N = 512, d = 16) →   of constant embeddings →   of N(0, 9 I) (std 3) →
2. Of N(0, I) after a random rotation →   after shifting the mean by 1 →
3. EMA with m = 0.99: after how many steps has the target moved ~63% of the way to a (frozen) online model? →
4. If the target encoder receives gradients (no stop-grad) but has EMA: does collapse come back? →

## Misses
-
