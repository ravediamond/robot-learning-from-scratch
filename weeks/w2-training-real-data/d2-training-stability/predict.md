# Predict BEFORE running `predict_run.py`

1. Fresh `nn.Linear(64, 100)` classifier on random inputs: cross-entropy at step 0 ≈ →
2. Targets `y = 50 + randn` (not normalized), model predicts ≈ 0 at init: MSE at step 0 ≈ →
3. `torch.nn.utils.clip_grad_norm_(params, 1.0)` returns: the norm before or after clipping? →
4. Grads are `[3, 4]` and `[0, 12]` (two params). Global norm → ; after clipping to 1.0, first grad becomes →
5. `torch.exp(torch.tensor(100.0))` in float32 →   in float64 →
6. `torch.log(torch.softmax(torch.tensor([0.0, 200.0]), 0))` →

## Misses
-
