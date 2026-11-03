# W7 D1: Representation collapse, and VICReg

Primer: `knowledge/05-jepa-ssl.md`. Book: 12.2 (collapse and the three ways to prevent it).

## The trap
Two views of the same input (two augmentations, or frame t and frame t+1) should map to nearby
embeddings: `loss = ||f(x1) − f(x2)||²`. The encoder finds the trivial solution immediately:
f(x) = constant. Zero loss, zero information. You will watch it happen today.

## VICReg (Bardes, Ponce, LeCun 2022)
```
invariance  = MSE(z1, z2)
variance    = mean over dims of relu(1 − sqrt(Var_batch(z_j) + eps))      for z1 and z2 (sum)
covariance  = sum of squared OFF-diagonal entries of Cov(z) / d             for z1 and z2 (sum)
loss = 25 · invariance + 25 · variance + 1 · covariance
```
Variance keeps every dimension alive (std ≥ 1 across the BATCH). Covariance decorrelates the
dimensions so they don't all encode the same thing.

**Effective rank** (to measure partial collapse): singular values s of the centred embedding
matrix, p = s / sum(s), erank = exp(−Σ p log p). Between 1 (one direction) and d.

## Today's build (`w7d1_vicreg.py`)
`vicreg_loss`, `effective_rank`, `train_siamese(loss_kind)` on 2-D toy points with noise
augmentations. Test: invariance-only collapses; VICReg does not.

Run: `uv run pytest weeks/w7-jepa-world-models/d1-collapse-vicreg`
