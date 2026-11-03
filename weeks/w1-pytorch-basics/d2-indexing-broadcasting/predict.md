# Predict BEFORE running `predict_run.py`

1. `(torch.zeros(3, 1) + torch.zeros(4)).shape` →
2. `(torch.zeros(2, 3) + torch.zeros(3, 2))`: shape or error? →
3. `(torch.zeros(5, 1, 6) - torch.zeros(1, 7, 6)).shape` →
4. `x = torch.arange(12).reshape(3, 4)`; `x[[0, 2], [1, 3]]` →
5. Same `x`; `x[:, [1, 3]].shape` →
6. Same `x`; `x[x > 5]` →
7. `torch.einsum("ij->j", torch.ones(3, 4))` →
8. `a = torch.zeros(3)`; `b = a`; `b += 1`; `a` →
9. `a = torch.zeros(3)`; `b = a`; `b = b + 1`; `a` →
10. `torch.ones(2, 3).sum(dim=(0, 1))` →

## Misses (one line each)
-
