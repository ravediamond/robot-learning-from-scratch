# Predict BEFORE running `predict_run.py`

Write your answer next to each `→`. Then run `uv run python predict_run.py`.
For each miss, write one line: what did I believe, what is true.

1. `torch.tensor([1, 2, 3]).dtype` →
2. `torch.tensor([1.0, 2, 3]).dtype` →
3. `x = torch.zeros(4, 50, 6)`; `x.mean(dim=1).shape` →
4. Same `x`; `x.mean(dim=1, keepdim=True).shape` →
5. `x.transpose(1, 2).shape` →
6. `x.transpose(1, 2).view(4, -1)`: works or error? →
7. `torch.stack([torch.zeros(6), torch.zeros(6)]).shape` vs `torch.cat([...]).shape` →
8. `torch.arange(10).unfold(0, 3, 1).shape` →
9. `(torch.tensor([1, 2]) / 2).dtype` →
10. `torch.tensor([1.0], dtype=torch.float16) + torch.tensor([1e-4], dtype=torch.float16)` still equals exactly 1.0 in float16? (yes/no) →

## Misses (one line each)
-
