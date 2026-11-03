# Predict BEFORE running `predict_run.py`

1. Parameters of `nn.MultiheadAttention(64, 8)` (with biases) →
2. Same with 1 head instead of 8 →
3. `x = torch.arange(24).view(1, 2, 12)`; `x.view(1, 2, 3, 4).transpose(1, 2)[0, 0]` →
4. Same x; the WRONG split `x.view(1, 3, 2, 4)[0, 0]` →
5. Cross-attention with 50 action tokens (queries) over 300 VLM tokens (keys): shape of the attention matrix per head →

## Misses
-
