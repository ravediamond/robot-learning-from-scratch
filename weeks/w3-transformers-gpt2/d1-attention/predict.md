# Predict BEFORE running `predict_run.py`

1. `torch.softmax(torch.tensor([1.0, 2.0, 3.0]) * 10, 0)` roughly →   and with `* 0.1` →
2. q, k with i.i.d. N(0,1) entries, d = 256: std of q·k →   of q·k/√d →
3. `torch.tril(torch.ones(3, 3, dtype=torch.bool))` →
4. Causal attention: output at position 0 equals →
5. softmax of a row that is all `-inf` →
6. Attention without any position information: shuffle the input tokens. What happens to the outputs? →

## Misses
-
