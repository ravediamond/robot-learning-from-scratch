# Predict BEFORE running `predict_run.py`

1. 256 bins over [−1, 1]: max decoding error →
2. A 50-step chunk that is a straight line in time: how many DCT coefficients per dimension carry most of the energy? →
3. A chunk with a sudden gripper close (a step from 1 to 0 at t = 20): does the DCT need few or many coefficients? →
4. Tokens per chunk, H = 50, D = 6: binning →   DCT keeping 8 coefficients per dim →

## Misses
-
