# Predict BEFORE running `predict_run.py`

1. 224×224 RGB image, 16×16 patches: number of tokens →   length of each flattened patch →
2. Same with 14×14 patches (SigLIP-style) →
3. RoPE rotation changes the norm of a vector? →
4. A 12-block pre-norm transformer at init: std of the residual stream after 12 blocks vs at the input (bigger, same, smaller)? →
5. Sinusoidal PE: `PE[t] · PE[t + k]` depends on t? →

## Misses
-
