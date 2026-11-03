# W2 D1: Dataset, DataLoader, episode windows

Book: P2.8, 10b (the data pipeline). Coding list item 20.

## The one idea

A robot dataset is a list of **episodes**. A training sample is a **window**:
the observation at time t and the next H actions `a[t : t+H]`. Near the end of an episode
the window runs past the last frame: pad it and return a mask. LeRobot does exactly this
(`action_is_pad`). The policy loss then ignores padded steps (W1 D2 review).

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self): ...             # number of samples
    def __getitem__(self, i): ...      # one sample (dict of tensors)

loader = DataLoader(ds, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(0),
                    collate_fn=..., num_workers=0, drop_last=False)
```
- `generator=` makes the shuffle reproducible. Without it, two runs see different orders.
- `collate_fn` turns a list of samples into a batch. The default stacks tensors and recurses
  into dicts; write it once by hand so you know.
- Global index → (episode, t): keep the cumulative lengths and use `torch.searchsorted`
  (or `bisect`). O(log E) per lookup, no per-frame Python list.

## Today's build (`w2d1_data.py`)
1. `EpisodeWindowDataset`: one sample per frame; returns `obs (Do,)`, `actions (H, Da)`,
   `action_mask (H,)`, `episode (int)`. Past the end: repeat the LAST action, mask False.
2. `collate_windows`: list of samples → batch dict, by hand.
3. `make_loader`: seeded, reproducible shuffle.
4. `task_balanced_weights`: per-sample weights so each task is sampled equally
   (use with `WeightedRandomSampler`). Your bowls set has 2 tasks.

Run: `uv run pytest weeks/w2-training-real-data/d1-datasets-loaders`
