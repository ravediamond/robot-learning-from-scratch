# Predict BEFORE running `predict_run.py`

1. u = 0, μ = 0, σ = 1 (one action dim). log N(u) = →   tanh correction = →   log π(a) = →
2. u = 3 (a ≈ 0.995). The correction term −log(1 − tanh²(u)) is (large positive / large negative / ≈ 0)? →
3. τ = 0.005. After how many updates has a target net moved ~63% of the way to a (fixed) critic? →
4. Why the minimum of two critics and not the mean? →
5. SAC on Reach2D after ~1,500 updates, deterministic evaluation: mean return ≈ → (random −21, expert −3)

## Misses
-
