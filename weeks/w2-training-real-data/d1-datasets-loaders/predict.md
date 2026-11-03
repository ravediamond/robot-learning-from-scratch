# Predict BEFORE running `predict_run.py`

1. 3 episodes of lengths 10, 20, 30, one sample per frame. `len(dataset)` →
2. Same dataset, `DataLoader(batch_size=16, drop_last=False)`: number of batches →  With `drop_last=True` →
3. Default collate on a list of 4 dicts `{"x": tensor of shape (3,)}`: type and shape of the batch →
4. Two `DataLoader(shuffle=True)` WITHOUT a generator, after `torch.manual_seed(0)` once at the top: same first batch twice? →
5. `torch.searchsorted(torch.tensor([10, 30, 60]), torch.tensor([0, 9, 10, 59]), right=True)` →
6. Weighted sampling with weights 1/count: task A has 90 samples, task B 10. Expected fraction of B in a batch →

## Misses (one line each)
-
