# Predict BEFORE running `predict_run.py`

1. A 64×64 weight matrix, values around 0.02, one row scaled ×100. Per-tensor int8: relative error on the normal rows? Per-channel? →
2. Activations: 10,000 values ~N(0, 1) plus 5 values at 80. Min-max scale vs 99.9th-percentile scale? →
3. `torch.round(x)` in a training graph: what is d(round(x))/dx? →
4. Model weights 4 bytes each in fp32. Memory saving with int8 weights? Speed-up on a CPU without int8 kernels? →

## Misses
-
