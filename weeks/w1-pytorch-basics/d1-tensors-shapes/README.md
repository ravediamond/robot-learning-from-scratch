# W1 D1: Tensors and shapes

Book: P2.1, P2.2 (`reference/journal/book/00p2-pytorch-for-the-coding-round.md`). Coding list items 05, 20 (start).

## The one idea

A tensor is a block of numbers + a **shape** + a **dtype** + a **device**. 90% of bugs in
robot-learning code are shape bugs that do not crash: the code runs, the numbers are wrong.

Robot data has a fixed vocabulary of shapes. Learn to say them:

| Symbol | Meaning | SO-101 example |
|---|---|---|
| `B` | batch (independent samples) | 32 |
| `T` | time steps in a window or episode | 50 |
| `H` | action chunk horizon | 50 for ACT/SmolVLA |
| `D` | feature dim (joints for state/action) | 6 (5 joints + gripper) |
| `C, Hh, W` | image channels, height, width | 3, 224, 224 |

## API you need today (no more)

```python
torch.tensor(data, dtype=torch.float32)   # from Python lists; ints default to int64!
x.shape, x.dtype, x.device
x.reshape(B * T, D)    # always works (copies if needed)
x.view(B * T, D)       # fails if memory is not contiguous (e.g. after .transpose)
x.transpose(0, 1), x.permute(1, 0, 2)
x.unsqueeze(0), x.squeeze(-1)
torch.stack([a, b], dim=0)   # new dim
torch.cat([a, b], dim=0)     # existing dim
x.mean(dim=0), x.std(dim=0), keepdim=True
x.unfold(dim, size, step)    # sliding windows, a view (no copy)
torch.zeros(...), torch.ones(...), torch.full(...)
```

**`dim=k` means "collapse dimension k"**. `x.mean(dim=0)` on `(B, D)` gives `(D,)`: one mean
per joint, averaged over the batch.

## Today's build (`w1d1_tensors.py`)

1. `to_tensor`: lists → float32 tensor.
2. `flatten_time` / `unflatten_time`: `(B, T, D) ↔ (B*T, D)`. Every per-frame MLP does this.
3. `pad_episodes`: episodes have different lengths. Pad to `(N, T_max, D)` and return a
   boolean mask `(N, T_max)`, True where real. Every dataloader for episodes does this.
4. `normalize`: per-joint z-score.
5. `action_chunks`: `(T, D)` → `(T-H+1, H, D)`, all windows of `H` future actions.
   This is the target format for ACT, SmolVLA and every chunked policy you ran.
6. `pick_device`: `mps` on your Mac, else `cuda`, else `cpu`.

Run: `uv run pytest weeks/w1-pytorch-basics/d1-tensors-shapes`
