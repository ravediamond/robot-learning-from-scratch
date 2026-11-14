# Predict BEFORE running `predict_run.py`

1. Each STEP is expert (25%), random (25%) or anti-expert (50%). Top-30% advantage steps: mostly from which source? →
2. Fraction of EXPERT steps with I = 1 →       of anti-expert steps →
3. Mean return when conditioning on I = 1 →   on I = 0 →   plain BC →   (expert −3)
4. Why keep the I = 0 data instead of filtering it out? →
5. n-step advantage with n = 1 vs n = 5 here: which labels the expert steps more cleanly, and why? →
6. One global threshold over two tasks, one with returns around −5 and one around −50. What goes wrong? →

## Misses
-
