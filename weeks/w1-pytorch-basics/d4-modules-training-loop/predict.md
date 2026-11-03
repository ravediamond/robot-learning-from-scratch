# Predict BEFORE running `predict_run.py`

1. Number of parameters in `nn.Linear(10, 20)` →
2. In `nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 6))` →
3. A module storing its layers in a plain Python list `self.layers = [nn.Linear(4, 4)]`: number of parameters reported by `.parameters()` →
4. `nn.Dropout(0.5)` on `torch.ones(8)` in train mode: what do the surviving values equal? →
5. Same, in eval mode →
6. MSE at step 0 of a fresh MLP (default init) predicting targets `y ~ N(0, 1)`: roughly →
7. Cross-entropy at step 0 for 10 classes with a fresh classifier: roughly →
8. `opt = torch.optim.SGD(model.parameters(), lr=0.1)`; you then replace `model.fc = nn.Linear(...)`. Does `opt` update the new layer? →

## Misses (one line each)
-
