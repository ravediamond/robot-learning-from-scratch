# Predict BEFORE running `predict_run.py`

1. Parameters in GPT-2 small's token embedding (50257 × 768) →  share of the total 124.4M →
2. Parameters per transformer block (D = 768) →
3. Loss at step 0 of a fresh GPT-2 (V = 50257) →
4. Memory of the fp32 weights of GPT-2 small →   of the Adam state on top →
5. Without weight tying, how many parameters? →

## Misses
-
