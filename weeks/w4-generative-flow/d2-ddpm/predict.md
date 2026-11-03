# Predict BEFORE running `predict_run.py`

1. Linear betas 1e-4 → 0.02, T = 100: abar at the last step ≈ →   T = 1000 →
2. At t = T−1 (T = 1000), is x_t mostly data or mostly noise? →
3. A denoiser that always outputs 0 (no information): its loss ≈ →
4. Sampling with T = 100 calls the network how many times per sample? →  Diffusion Policy at T = 100 in a 10 Hz control loop on an Orin: feasible? →

## Misses
-
