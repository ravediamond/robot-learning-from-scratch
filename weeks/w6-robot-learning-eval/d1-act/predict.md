# Predict BEFORE running `predict_run.py` (after your tests pass; ~6 min, it trains 6 models)

The script trains ACT on the bimodal obstacle demos for beta in {0.01, 0.05, 1.0} and 2 seeds each,
and reports collision rates with z = 0 and with z ~ N(0, I).

1. beta = 1.0 (KL dominates): does z carry information? Collision rate with z = 0 →
2. Is collision with z = 0 closer to the W4 MSE head (~50%) or the flow head (~6%)? Why? →
3. How different will the two seeds of the same beta be? (±2 points? ±20?) →
4. Which conclusion would you have drawn from ONE run, and would it survive the second seed? →

## Misses
-
