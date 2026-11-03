"""Run AFTER writing your guesses in predict.md."""
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.utils.data import default_collate

ds = TensorDataset(torch.arange(60.0))
print("1 len:", len(ds))
print("2 batches:", len(DataLoader(ds, batch_size=16)), "drop_last:", len(DataLoader(ds, batch_size=16, drop_last=True)))
b = default_collate([{"x": torch.zeros(3)} for _ in range(4)])
print("3 collate:", type(b).__name__, {k: tuple(v.shape) for k, v in b.items()})
torch.manual_seed(0)
a = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))[0]
c = next(iter(DataLoader(ds, batch_size=4, shuffle=True)))[0]
print("4 same first batch:", torch.equal(a, c), a.tolist(), c.tolist())
print("5 searchsorted:", torch.searchsorted(torch.tensor([10, 30, 60]), torch.tensor([0, 9, 10, 59]), right=True).tolist())
w = torch.cat([torch.full((90,), 1 / 90), torch.full((10,), 1 / 10)])
idx = torch.tensor(list(WeightedRandomSampler(w, 10000, generator=torch.Generator().manual_seed(0))))
print("6 fraction of task B:", round((idx >= 90).float().mean().item(), 3))
print("\n4: the global seed is consumed by the first loader; the second starts from a different RNG state.")
print("Pass generator=torch.Generator().manual_seed(s) to make each loader reproducible on its own.")
