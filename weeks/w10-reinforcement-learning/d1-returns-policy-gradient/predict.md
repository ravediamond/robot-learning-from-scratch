# Predict BEFORE running `predict_run.py`

1. rewards [1, 1, 1], γ = 0.9, no done. G_0, G_1, G_2 →
2. Same rewards, done = [F, T, F] (an episode ends at step 1). G_0 →
3. GAE with λ = 1 equals →          with λ = 0 equals →
4. Rewards are all −1 per step (a 20-step episode). Without a baseline, every action's log-prob is pushed (up / down)? Does learning still work in expectation? →
5. Variance of the REINFORCE gradient with a baseline vs without, on Reach2D: ratio ≈ →

## Misses
-
