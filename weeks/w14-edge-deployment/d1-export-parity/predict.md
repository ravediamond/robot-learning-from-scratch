# Predict BEFORE running `predict_run.py`

1. A policy is exported without its normalisation stats; joint positions are in radians (std ≈ 0.5), the model expects z-scores. What does the arm do? →
2. Exported with an example batch of 1 and no dynamic axes. You call it with a batch of 2. →
3. Max absolute error between PyTorch fp32 and ONNX Runtime fp32 on the same input: order of magnitude? →
4. The model has dropout and you forget `model.eval()` before comparing outputs. What does the parity check say? →

## Misses
-
