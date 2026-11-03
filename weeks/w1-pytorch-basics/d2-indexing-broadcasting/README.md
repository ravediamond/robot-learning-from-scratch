# W1 D2: Indexing, broadcasting, einsum

Book: P2.1 (broadcasting), P2.15. Coding list items 05 (masked losses), 20.

## The one idea

**No Python loops over the batch.** Interviewers watch for it, and on a GPU a loop is 100×
slower. Three tools replace loops:

### 1. Broadcasting
Shapes are aligned **from the right**. Each pair of dims must be equal, or one of them 1.
```
(N, 1, D)
(1, M, D)
--------- 
(N, M, D)     every row of a against every row of b
```
`a[:, None, :]` is the same as `a.unsqueeze(1)`: insert a size-1 dim to make broadcasting do the loop.

### 2. Advanced indexing and `gather`
```python
x[torch.arange(B), idx]          # (B,): row i, column idx[i]
x.gather(dim=1, index=idx[:, None]).squeeze(1)   # same thing, generalizes to more dims
```
Rule of `gather`: `index` has the same number of dims as `x`; output has the shape of `index`.

### 3. Masks
Bool tensors select or zero things. For a masked mean, **divide by the count of valid
items, not by the length**. Padding must contribute exactly zero, and must not dilute the mean.

### 4. einsum
Write the shapes, name the dims, let einsum do the sum over letters that disappear:
```python
torch.einsum("bij,bj->bi", W, x)   # batched matrix @ vector
torch.einsum("bhtd,bhsd->bhts", q, k)   # attention scores (week 3)
```

## Today's build (`w1d2_broadcast.py`)

1. `pairwise_sq_dist`: (N, D), (M, D) → (N, M). Used in nearest-neighbor baselines and kNN policies.
2. `select_per_row`: pick one entry per row (`gather`). Used in cross-entropy and Q-learning.
3. `masked_mean`: average over valid time steps only.
4. `batched_matvec`: with einsum.
5. `best_of_n`: given N candidate action chunks per sample and a score each, return the
   best chunk per sample. This is exactly the best-of-N reranking you tested (H0) and abandoned.
   You should be able to write it in 3 lines.

Run: `uv run pytest weeks/w1-pytorch-basics/d2-indexing-broadcasting`
