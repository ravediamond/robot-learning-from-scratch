# Predict BEFORE running `predict_run.py`

1. `x = torch.tensor(3.0, requires_grad=True)`; `y = x ** 2`; `y.backward()`; `x.grad` →
2. Same, but call `y = x ** 2; y.backward()` twice in a row (rebuilding y each time). `x.grad` →
3. `y = x ** 2`; `y.backward()`; `y.backward()` again on the SAME y: value or error? →
4. `x = torch.tensor([1.0, 2.0], requires_grad=True)`; `(x * 2).backward()`: value or error? →
5. `x = torch.tensor(2.0, requires_grad=True)`; `z = (x * 3).detach() * x`; `z.backward()`; `x.grad` →
6. `torch.tensor(2, requires_grad=True)` (an int): value or error? →
7. `a = torch.tensor(1.0, requires_grad=True)`; `b = a * 2`; `b.is_leaf`, `a.is_leaf`, `b.grad` after `b.backward()` →
8. Inside `with torch.no_grad(): y = x * 2`, `y.requires_grad` →
9. `x = torch.tensor(1.0, requires_grad=True)`; `y = torch.relu(x - 2)`; `y.backward()`; `x.grad` →

## Misses (one line each)
-
