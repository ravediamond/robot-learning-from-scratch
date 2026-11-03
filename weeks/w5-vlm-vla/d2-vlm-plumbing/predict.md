# Predict BEFORE running `predict_run.py`

1. 512×512 image, patch 16, pixel shuffle r = 4: tokens before →   after →   channels after (C = 768) →
2. SmolVLA-style input: 2 cameras × 64 tokens, 20 text tokens, 1 state, 50 actions. Total length →
3. With block masks [148 prefix | 1 state | 50 actions], how many query-key pairs are allowed (actions bidirectional) →   vs full causal mask over 199 tokens →
4. Does the prefix output change when the action tokens change? →

## Misses
-
