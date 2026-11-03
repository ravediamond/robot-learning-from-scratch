# Predict BEFORE running `predict_run.py` (after your tests pass)

1. Random actions, 25 steps: success (within 0.15 of the goal) →
2. CEM with latent-L2 cost →   with probe-decoded cost →
3. Spearman correlation between latent distance and true distance, for random pairs of positions →
4. Surprise AUROC for teleports →   for a SMALL teleport (0.1 instead of 0.6) →

## Misses
-
