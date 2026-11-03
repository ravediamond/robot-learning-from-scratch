# Chapter P2. PyTorch for the coding round

**If you are new to deep learning, read chapter P1 first (`00p1-deep-learning-from-zero.md`).** It explains every concept this chapter uses (loss, gradient, backpropagation, layers, attention, JEPA, CEM, flow matching, the statistics) with small worked examples, and ends with a vocabulary table. This chapter assumes those ideas and shows how they are written in PyTorch.

The coding list (`exercises/coding/README.md`) asks you to write 24 components cold. This chapter gives you every concept and every PyTorch primitive those components are built from, with the mistakes that usually go with them. Sections P2.1–P2.15 give the concepts and APIs, not the solutions. Two worked builds at the end, P2.19 (an autograd engine from scratch) and P2.20 (a tiny GPT, end to end), show complete working code for the canonical items, so that nothing outside this book is needed; read them once, run them, then close the book and write the items cold. Each section ends with the exercise numbers it serves.

Every code block in this chapter was run against PyTorch 2.14 on CPU on 2026-10-24, top to bottom in one Python session. Run them yourself the same way: open a REPL, paste section by section, and change things until each line's output stops surprising you.

```python
import math, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
```

## P2.1 Tensors: shape is the whole game

A tensor is an n-dimensional array (P1.5) with a `shape` (the size of each axis), a `dtype` (the number format: float32, int64, bool…) and a `device` (CPU or a GPU). In the interview, almost every bug is a shape bug, and the habit that prevents them is to **say the shape out loud on every line** and write it in a comment.

```python
x = torch.randn(2, 3, 4)            # (B, T, D): batch 2, time 3, features 4
x.shape, x.dtype, x.device          # torch.Size([2, 3, 4]), float32, cpu
x.ndim, x.numel()                   # 3, 24
torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 7.0), torch.arange(5), torch.linspace(0, 1, 5)
torch.zeros_like(x), torch.randn_like(x)   # same shape/dtype/device as x
torch.tensor([1, 2, 3])             # int64 by default; torch.tensor([1., 2.]) is float32
```

**Indexing** follows NumPy. `x[0]` drops the first axis; `x[:, -1]` is the last time step of every batch element, shape (B, D); `x[..., 0]` is the first feature everywhere, shape (B, T). `None` inserts an axis: `x[:, None]` is (B, 1, T, D).

**Reshaping.** Four operations, four different meanings:

| Op | What it does | Copies? |
|---|---|---|
| `view(*shape)` | reinterpret the same memory with a new shape | never; fails if memory is not contiguous |
| `reshape(*shape)` | like `view` but copies if it must | sometimes |
| `transpose(a, b)` / `permute(*dims)` | reorder axes | never; result is **non-contiguous** |
| `unsqueeze(d)` / `squeeze(d)` | add / remove a size-1 axis | never |

```python
B, T, H, Dh = 2, 5, 4, 8             # batch, time, heads, head dim
y = torch.randn(B, T, H * Dh)        # (2, 5, 32)
y = y.view(B, T, H, Dh)              # (2, 5, 4, 8): split the last axis into heads
y = y.transpose(1, 2)                # (2, 4, 5, 8): heads before time
y.is_contiguous()                    # False
y = y.transpose(1, 2).contiguous().view(B, T, H * Dh)   # back to (2, 5, 32)
```

The split-heads / merge-heads pair above is half of exercise 02. The classic error is calling `.view` after `.transpose`; it raises, and the fix is `.contiguous()` first, or `.reshape`.

**Combining.** `torch.cat([a, b], dim=d)` joins along an existing axis (sizes must match elsewhere). `torch.stack([a, b], dim=d)` creates a new axis. Stacking a list of K tensors of shape (B, D) along dim 1 gives (B, K, D).

**Broadcasting.** Two shapes are aligned from the right; each pair of sizes must be equal or one of them must be 1; size-1 axes are stretched. (B, T, D) + (D,) works (bias per feature). (B, T, D) * (B, T, 1) works (a per-step scale). (B, T, D) + (B, D) does **not** work: right-aligned, T meets B. Write `x + b[:, None, :]` to make it (B, 1, D).

```python
a = torch.randn(4, 3)                # (N, D)
mu = a.mean(dim=0)                   # (D,)
a_centered = a - mu                  # (N, D) - (D,) → broadcast over N
pair = a[:, None, :] - a[None, :, :] # (N, 1, D) - (1, N, D) → (N, N, D): all pairwise differences
pair.shape
```

The pairwise-difference trick is how you compute distance matrices without loops. Watch memory: (N, N, D) grows fast.

**Reductions.** `sum`, `mean`, `max`, `min`, `std`, `var`, `norm`, `logsumexp` all take `dim=` and `keepdim=`. `keepdim=True` keeps the reduced axis as size 1, which is what you want when you broadcast the result back.

```python
z = torch.randn(3, 5)
z.mean(dim=-1).shape, z.mean(dim=-1, keepdim=True).shape   # (3,), (3, 1)
z.max(dim=-1)                        # returns (values, indices)
z.var(dim=-1, unbiased=False)        # population variance: what LayerNorm uses
z.norm(dim=-1)                       # L2 norm per row
```

**einsum** names the axes and states the contraction. It is the clearest way to write attention scores and anything batched.

```python
q = torch.randn(2, 4, 5, 8)          # (B, H, Tq, Dh)
k = torch.randn(2, 4, 6, 8)          # (B, H, Tk, Dh)
scores = torch.einsum('bhqd,bhkd->bhqk', q, k)            # (2, 4, 5, 6)
torch.allclose(scores, q @ k.transpose(-2, -1))           # True: same thing with matmul
```

`@` (matmul) on tensors with more than two axes treats the leading axes as batch and multiplies the last two.

**Selecting.** `torch.where(cond, a, b)` picks elementwise. `x.masked_fill(mask, value)` writes value where mask is True. `torch.gather(x, dim, idx)` picks one entry per row by index (cross-entropy by hand uses it). `topk(k, dim)` returns the k largest values and their indices (CEM elites use it). `argsort` gives the permutation that sorts.

```python
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])  # (N=2, C=3)
target = torch.tensor([0, 2])                                # (N,) class indices, int64
picked = logits.gather(1, target[:, None]).squeeze(1)        # (N,): logit of the true class
vals, idx = logits.topk(2, dim=1)                            # two largest per row
torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)   # True above the diagonal: the future
```

The last line is the causal mask of exercise 01: position i must not see positions j > i, which are exactly the entries above the diagonal.

*Serves: 01, 02, 04, 07, 13, 15, 17, 19, 20, 23.*

## P2.2 dtypes, devices, numerics

**dtypes.** Default float is `float32`. Integer indices (class targets, `gather` indices, embedding ids) must be `int64` (`torch.long`). Masks are `torch.bool`. Mixing a float tensor with an int tensor promotes to float; mixing float32 with float64 promotes to float64 (and then `assert_close` against a float32 reference fails on dtype).

| dtype | Bits | Exponent range | Precision | Use |
|---|---|---|---|---|
| float32 | 32 | wide | ~7 digits | default, reference checks |
| bfloat16 | 16 | **same range as fp32** | ~3 digits | training on recent GPUs (H100, A100, Orin's Ampere) |
| float16 | 16 | narrow (max 65 504) | ~3 digits | needs loss scaling, overflows easily |
| float64 | 64 | very wide | ~16 digits | `gradcheck` only |

Why bf16 over fp16 (a quiz question): same exponent bits as fp32, so no overflow and no loss scaling; the price is fewer mantissa bits, which is why losses and reductions are usually kept in fp32.

**devices.** `x.to('cuda')`, `x.cuda()`, `x.cpu()`, `model.to(device)`. Every tensor in an operation must be on the same device. Write device-agnostic code: `device = 'cuda' if torch.cuda.is_available() else 'cpu'` and create new tensors with `device=x.device` or `torch.zeros_like(x)`. Moving a tensor to the GPU inside the loop that the model already lives on is the most common runtime error in a live round.

**Numerical stability.** Three rules that appear in almost every exercise:

1. **Softmax.** `exp` overflows for large logits. Subtract the row max first; it does not change the result because softmax is shift-invariant. `F.softmax` and `torch.logsumexp` already do this.
2. **Log of a softmax.** Never `torch.log(F.softmax(x))`: small probabilities underflow to 0 and the log is −inf. Use `F.log_softmax(x, dim)`, which computes `x − logsumexp(x)`.
3. **Masking before softmax.** Set masked positions to `-inf` (or `torch.finfo(dtype).min`), not to 0. A 0 logit still gets weight exp(0) = 1. If a whole row is masked, softmax of all −inf is NaN; make sure every query can see at least itself.

```python
s = torch.tensor([1000.0, 1001.0, 1002.0])
torch.exp(s)                                           # inf inf inf
F.softmax(s, dim=0)                                    # fine: tensor([0.0900, 0.2447, 0.6652])
s - torch.logsumexp(s, dim=0)                          # = F.log_softmax(s, 0)
m = torch.tensor([False, False, True])
F.softmax(s.masked_fill(m, float('-inf')), dim=0)      # last weight exactly 0
```

**eps.** Add a small epsilon wherever you divide by a norm or a standard deviation (`x / (x.norm(dim=-1, keepdim=True) + 1e-8)`, LayerNorm's `+ eps` inside the square root). Say where you put it and why.

*Serves: 01, 02, 04, 07, 11, 15, 24.*

## P2.3 Autograd

This section is P1.3 (backpropagation) in code. PyTorch records the operations applied to tensors that have `requires_grad=True` ("track gradients for this one"), building the computation graph: a record of which operation produced which tensor from which inputs. `loss.backward()` walks it backwards and writes ∂loss/∂leaf into each leaf's `.grad`. Five facts cover almost every interview question about it.

```python
w = torch.tensor([2.0, -1.0], requires_grad=True)   # a leaf: created by you, requires grad
x = torch.tensor([3.0, 4.0])                         # data: no grad
loss = ((w * x).sum() - 1.0) ** 2                   # scalar
loss.backward()
w.grad                                               # d/dw = 2 * (w·x - 1) * x = 2 * 1 * [3, 4] = [6., 8.]
```

1. **`backward()` needs a scalar** (or an explicit `gradient=` argument). If your loss has shape (B,), you forgot a `.mean()`.
2. **Gradients accumulate.** A second `backward()` adds to `.grad`. That is why the loop calls `optimizer.zero_grad()` every step, and why gradient accumulation (item 24) works by simply not zeroing for k steps.
3. **`detach()` cuts the graph.** `y.detach()` is the same values with no history: gradient does not flow through it. This is the **stop-gradient** of JEPA (exercise 14): the target branch's output is detached so the encoder is not trained to move the target towards the prediction, which is one of the routes to collapse.
4. **`torch.no_grad()`** turns recording off inside a block: evaluation, EMA updates, anything you never differentiate. `torch.inference_mode()` is a stricter, faster version for pure inference. Neither changes dropout or batch-norm behaviour; that is `model.eval()` (P2.4).
5. **The graph is freed after `backward()`.** Calling `backward()` twice on the same graph raises unless you passed `retain_graph=True`. If you hit this, you are usually reusing a tensor across iterations that should have been recomputed or detached.

```python
target = torch.randn(3, requires_grad=True)
pred = torch.randn(3, requires_grad=True)
loss = F.mse_loss(pred, target.detach())             # stop-gradient on the target
loss.backward()
pred.grad is not None, target.grad is None           # (True, True): no grad reached the target
```

That last line is the check exercise 14 asks for.

**Leaves and non-leaves.** A *leaf* is a tensor at the start of the graph, one you created yourself rather than one computed from others: the parameters are leaves, intermediate results are not. Only leaves (tensors you created with `requires_grad=True`, and `nn.Parameter`s) keep `.grad` by default. An intermediate tensor's `.grad` is `None` unless you call `.retain_grad()` on it. If you want a gradient with respect to something without touching `.grad`, use `torch.autograd.grad(loss, [inputs])`.

**In-place operations** (`x += 1`, `x.relu_()`, anything ending in `_`) on a tensor that autograd saved for the backward pass raise "a leaf Variable that requires grad is being used in an in-place operation" or "one of the variables needed for gradient computation has been modified". Avoid in-place on anything in the graph; use it freely inside `torch.no_grad()` (that is how optimisers and EMA update parameters).

**Checking a hand-written backward** (exercise 08). Compare your gradients with autograd on the same inputs, and for full rigour use `gradcheck` in float64, which compares against finite differences.

```python
from torch.autograd import gradcheck
inp = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
W = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
gradcheck(lambda a, b: (a @ b.T).tanh().sum(), (inp, W))   # True
```

For a linear layer y = x Wᵀ + b with upstream gradient g = ∂L/∂y of shape (N, out): ∂L/∂W = gᵀ x (shape out × in), ∂L/∂b = g summed over N, ∂L/∂x = g W (shape N × in). Derive these once on paper from the index form y_ni = Σ_j x_nj W_ij + b_i; the interviewer will ask you to.

*Serves: 03, 06, 08, 14, 15, 22, the debugging drill.*

## P2.4 nn.Module: how models are organised

A module is PyTorch's unit of model-building: a Python class that holds parameters (the learned numbers of P1.1), may contain other modules (a model contains layers, a layer contains its weight and bias), and defines a `forward` method that computes the output from the input. You write `__init__` (create layers and parameters) and `forward` (use them). You call the module, `model(x)`, never `model.forward(x)`: the call runs hooks that `forward` skips.

```python
class TinyMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                           # forget this and nothing registers
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.scale = nn.Parameter(torch.ones(d_out)) # a learnable tensor
        self.register_buffer('step', torch.zeros(())) # state that is saved but not trained

    def forward(self, x):                            # x: (N, d_in)
        return self.fc2(F.gelu(self.fc1(x))) * self.scale   # (N, d_out)

net = TinyMLP(4, 16, 3)
sum(p.numel() for p in net.parameters())             # 4*16+16 + 16*3+3 + 3 = 134
[n for n, _ in net.named_parameters()]               # fc1.weight, fc1.bias, fc2.weight, fc2.bias, scale
list(net.state_dict().keys())                        # the above plus 'step' (buffers are saved too)
```

Things to know cold:

- **Registration.** Attributes that are `nn.Module` or `nn.Parameter` are registered automatically. A plain Python list of layers is **not**; use `nn.ModuleList`. A plain tensor attribute is not saved or moved by `.to(device)`; use `register_buffer`.
- **`train()` / `eval()`** flip `self.training` on every sub-module. Only layers whose behaviour depends on it change: `Dropout` (off in eval), `BatchNorm` (uses running statistics in eval). LayerNorm, Linear, attention without dropout behave identically. Forgetting `model.eval()` before evaluation is a planted bug in the debugging drill; forgetting `model.train()` after evaluation is the same bug in the other direction.
- **Freezing.** `p.requires_grad_(False)` for every parameter of a module freezes it; also leave it out of the optimiser. `module.eval()` alone does **not** freeze anything.
- **Copying for an EMA target.** `target = copy.deepcopy(online)`, then freeze the target's parameters. Two modules sharing one parameter object is a subtle bug; `deepcopy` avoids it.
- **`state_dict` / `load_state_dict`.** A dict from names to tensors. `load_state_dict(sd, strict=True)` raises on missing or unexpected keys and returns the lists when `strict=False`; exercise 22 is making the count of loaded tensors visible so a silent partial load (G11-A's random-init encoder) cannot happen again.

```python
sd = net.state_dict()
sd.pop('fc2.bias')
result = net.load_state_dict(sd, strict=False)
result.missing_keys, result.unexpected_keys          # (['fc2.bias'], [])
```

**Initialisation.** Why initialisation matters is in P1.9. `nn.Linear` initialises weights uniformly in ±1/√fan_in, where fan_in is the number of inputs to the layer (the scheme is called Kaiming-uniform, after its author, with a=√5) and biases in the same range. `nn.init.xavier_uniform_`, `nn.init.zeros_`, `nn.init.normal_(w, std=0.02)` exist when you want to override, inside `torch.no_grad()` or with the in-place `nn.init` functions, which handle that. For an action head that should start near zero output (residual policies), zero-init the last layer.

*Serves: 02, 07, 08, 09, 14, 22.*

## P2.5 The layers you compose

**`nn.Linear(in, out)`** stores `weight` of shape **(out, in)** and computes `x @ weight.T + bias`. It acts on the last axis, so (B, T, in) → (B, T, out) with no reshaping.

**`nn.LayerNorm(D)`** normalises over the last axis: subtract the mean, divide by √(var + eps) with the **biased** variance, then multiply by a learned `weight` (init 1) and add `bias` (init 0). **RMSNorm** skips the mean subtraction and the bias: x / √(mean(x²) + eps) · weight. Exercise 07 is writing both and matching `nn.LayerNorm` within 1e-5 (PyTorch also ships `nn.RMSNorm`).

**`nn.Embedding(V, D)`** is a lookup table: integer ids (any shape) → vectors (that shape + D). Learned query tokens and positional tables are embeddings or plain `nn.Parameter`s of shape (n, D).

**Activations.** `F.relu`, `F.gelu`, `F.silu`, `torch.tanh`, `torch.sigmoid`. Transformers use GELU; the predictor MLPs in the LeWM family do too.

**Dropout.** `nn.Dropout(p)` zeroes each element with probability p **and scales the survivors by 1/(1−p)** in training, so the expected value is unchanged and eval needs no rescaling.

**Attention.** Scaled dot-product attention: softmax(Q Kᵀ / √d_k + mask) V. Shapes: Q (B, H, Tq, Dh), K and V (B, H, Tk, Dh), scores (B, H, Tq, Tk), output (B, H, Tq, Dh). Divide by √d_k so the dot products have unit variance at init and softmax does not saturate (a likely follow-up question). Multi-head: one projection to 3·D (or three to D), split into H heads of Dh = D/H, attend per head, merge, output projection. PyTorch has both reference implementations you will test against:

```python
D, H = 32, 4
mha = nn.MultiheadAttention(D, H, batch_first=True)     # batch_first! default is (T, B, D)
xx = torch.randn(2, 5, D)
out, attn = mha(xx, xx, xx, need_weights=True)           # out (2, 5, 32), attn (2, 5, 5) averaged over heads
mha.in_proj_weight.shape, mha.out_proj.weight.shape      # (96, 32), (32, 32): q, k, v stacked in that order
qh = torch.randn(2, H, 5, D // H)
F.scaled_dot_product_attention(qh, qh, qh, is_causal=True).shape   # (2, 4, 5, 8)
```

To match `nn.MultiheadAttention` exactly (exercise 02), copy its weights into your module: `in_proj_weight` is the Q, K and V projection weights stacked along dim 0 (rows 0:D are Q, D:2D are K, 2D:3D are V), likewise `in_proj_bias`; `out_proj` is an ordinary Linear. Note the boolean `attn_mask` convention differs between the two functions: in `nn.MultiheadAttention` **True means "not allowed"**, in `F.scaled_dot_product_attention` **True means "allowed"**. Say this out loud if you use either; it is a classic silent bug.

*Serves: 01, 02, 07, 09, 14.*

## P2.6 Losses

The functional forms in `torch.nn.functional` are what you use in a live round.

- **`F.cross_entropy(logits, target)`**: `logits` (N, C) are the raw class scores of P1.6, **not** softmaxed; `target` (N,) int64 class indices. It is log-softmax plus negative log-likelihood, averaged. For sequences, flatten: logits (B, T, C) → (B·T, C), targets (B, T) → (B·T,). `ignore_index=-100` skips padded positions.
- **`F.binary_cross_entropy_with_logits(logit, y)`** for binary targets; takes raw logits and float targets. Never apply sigmoid first and then `F.binary_cross_entropy`: less stable.
- **`F.mse_loss`, `F.l1_loss`, `F.smooth_l1_loss` / `F.huber_loss`** take `reduction='mean' | 'sum' | 'none'`.

**Masked losses** (exercise 05, and every episodic dataset with padded tails): compute with `reduction='none'`, multiply by the mask, divide by the number of valid elements, not by the total.

```python
pred = torch.randn(2, 4, 3); tgt = torch.randn(2, 4, 3)        # (B, T, A): action chunks
mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32)   # (B, T): 1 = real step
per = F.mse_loss(pred, tgt, reduction='none').mean(-1)        # (B, T): mean over action dims
loss = (per * mask).sum() / mask.sum().clamp(min=1)           # mean over valid steps only
```

Dividing by `mask.numel()` instead of `mask.sum()` silently down-weights short episodes, which is a planted-bug candidate.

**Cross-entropy by hand** (exercise 04): `-(log_softmax(logits)[range(N), target]).mean()`, using `gather` or advanced indexing; check against `F.cross_entropy` with `torch.testing.assert_close`.

*Serves: 03, 04, 05, 09, 11, 14, 20.*

## P2.7 Optimisers and schedules

**The update order** every step: `zero_grad` → forward → loss → `backward` → (clip) → `step` → (scheduler step). Getting this order wrong is the commonest training-loop bug.

```python
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
xb, yb = torch.randn(8, 4), torch.randint(0, 3, (8,))
opt.zero_grad(set_to_none=True)          # None instead of zeros: frees memory, and a missing grad shows up as None
loss = F.cross_entropy(net(xb), yb)
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)   # returns the norm before clipping: log it
opt.step()
```

**The update rules** you need for exercise 06, with g the gradient, θ the parameter, η the learning rate:

- **SGD with momentum** (PyTorch convention): v ← μ v + g; θ ← θ − η v.
- **Adam**: m ← β₁ m + (1−β₁) g; v ← β₂ v + (1−β₂) g²; m̂ = m / (1−β₁ᵗ); v̂ = v / (1−β₂ᵗ); θ ← θ − η m̂ / (√v̂ + ε). t counts steps starting at 1. The bias correction matters only early; forgetting it makes the first steps too small.
- **AdamW**: Adam plus decoupled weight decay θ ← θ − η λ θ applied directly, not through g. In plain `Adam`, `weight_decay` adds λθ to g and then gets rescaled by √v̂, which is why AdamW exists (a fundamentals question).

To write your own optimiser: a class holding the parameter list and per-parameter state (`m`, `v`, step count), a `zero_grad` that sets each `p.grad = None`, and a `step` that updates `p` **in place inside `torch.no_grad()`** (`p.add_(...)`, `p.mul_(...)`, or `p -= ...`). Then compare one step against `torch.optim.Adam` on cloned parameters and identical gradients.

**Parameter groups** let you give different learning rates or no weight decay to some parameters (biases, norms, embeddings): pass a list of dicts `[{'params': [...], 'weight_decay': 0.0}, {'params': [...]}]`.

**Schedules** (exercise 10). Linear warm-up from 0 to the peak over W steps, then cosine decay to a floor: lr(t) = peak · t/W for t < W; lr(t) = floor + ½(peak − floor)(1 + cos(π (t−W)/(T−W))) after. Implement it as a plain function first and test the three boundary values; then wrap it with `torch.optim.lr_scheduler.LambdaLR(opt, lambda t: f(t)/peak)`, which multiplies the optimiser's base lr by your factor. Call `scheduler.step()` once per optimiser step, after `opt.step()`.

*Serves: 03, 06, 09, 10, 24.*

## P2.8 Data: Dataset, DataLoader, splits

A `Dataset` has `__len__` and `__getitem__(i)` returning one sample (a tensor, a tuple or a dict). A `DataLoader` batches samples: `DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0, collate_fn=None)`. The default collate stacks tensors and recurses into tuples and dicts; write your own `collate_fn` only for variable-length items.

```python
from torch.utils.data import Dataset, DataLoader

class Windows(Dataset):
    """(obs_t, action_{t:t+K}) windows from a list of episodes; illustrates the API, not exercise 20's solution."""
    def __init__(self, episodes, K):
        self.episodes, self.K = episodes, K
        self.index = [(e, t) for e, ep in enumerate(episodes) for t in range(len(ep['obs']))]
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        e, t = self.index[i]
        ep = self.episodes[e]
        return {'obs': ep['obs'][t], 'act': ep['act'][t:t + self.K]}   # tail windows are shorter: exercise 20 pads them

eps = [{'obs': torch.randn(10, 6), 'act': torch.randn(10, 6)} for _ in range(3)]
ds = Windows(eps, K=4)
len(ds), ds[0]['act'].shape                         # (30, torch.Size([4, 6]))
```

The default collate fails on the last windows of each episode because their `act` is shorter than K; padding plus a mask (exercise 20) is the fix.

**Split by episode, never by frame.** Neighbouring frames are near-duplicates; a frame-level split leaks test information into training and inflates every metric. Choose episode ids for each split first (with a fixed seed), then build windows only from those episodes. Assert the two id sets are disjoint.

**Normalisation statistics** (exercise 21) are computed on the **training episodes only**, saved with the checkpoint, and applied unchanged at validation and on the robot. Computing them on everything leaks; recomputing them at inference on live data silently changes the input distribution.

**Reproducible shuffling.** Pass a seeded `torch.Generator()` to the DataLoader (`generator=g`), and seed workers with `worker_init_fn` if `num_workers > 0`.

*Serves: 09, 16, 20, 21, the debugging drill (unshuffled sorted data, wrong-split statistics).*

## P2.9 The training loop, as a checklist

Exercise 03 is yours to write; this is the list it must satisfy.

1. Seed everything (P2.10). Build model, optimiser, loaders; move the model to the device **once**.
2. For each epoch: `model.train()`; for each batch, move it to the device, then `zero_grad` → forward → loss → `backward` → clip (optional) → `step` → scheduler.
3. Log the loss as a Python float with `loss.item()`, not the tensor (keeping tensors in a list keeps their graphs alive and leaks memory).
4. Evaluation: `model.eval()` and `torch.no_grad()`, accumulate the loss **weighted by batch size** (or by valid-mask count), divide by the total; then `model.train()` again.
5. **Overfit one batch** before anything else: the same batch for 200 steps must drive the training loss close to zero. If it does not, the bug is in the model, the loss or the target, not in the hyper-parameters.
6. Save a checkpoint (P2.12) at the end and at the best validation loss.

Things they watch for: the order in step 2, `eval()`/`no_grad()` both present in step 4, `.item()` in step 3, and whether you ran step 5 before declaring the loop done.

*Serves: 03, 09, the debugging drill.*

## P2.10 Reproducibility

```python
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)   # torch.manual_seed also seeds CUDA
seed_all(0); a1 = torch.randn(3)
seed_all(0); a2 = torch.randn(3)
torch.equal(a1, a2)                                      # True
```

For bit-for-bit determinism on GPU you additionally need `torch.use_deterministic_algorithms(True)` (raises on ops that have no deterministic version) and, for cuBLAS, the environment variable `CUBLAS_WORKSPACE_CONFIG=:4096:8`; expect a slowdown. In practice: seed everything, log the seed, and report the variance across seeds rather than claiming determinism. Chapter 7's 26-point seed floor is why.

Seed **before** splitting the data, not after: a seed set after the split leaves the split itself random (a planted bug).

*Serves: 03, 12, 17, 24.*

## P2.11 Mixed precision

```python
lin = nn.Linear(8, 8)
with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
    out = lin(torch.randn(4, 8))
out.dtype                                                # torch.bfloat16: matmuls run in bf16
```

`autocast` runs matmuls and convolutions in the low-precision type and keeps sensitive ops (softmax, norms, losses) in float32. Parameters stay float32 (the "master weights"); gradients come back in float32. With **bf16 no scaler is needed**. With **fp16** you need `torch.amp.GradScaler`: it multiplies the loss by a large factor so small gradients do not underflow, unscales before the optimiser step, and skips steps whose gradients overflowed. If you clip gradients under fp16, call `scaler.unscale_(opt)` before clipping.

Casting the **loss** itself to fp16 (a planted bug) loses precision exactly where it matters; compute the loss in float32.

*Serves: 24, the debugging drill.*

## P2.12 Checkpoints

A checkpoint is a dict of state dicts plus whatever you need to resume exactly:

```python
ckpt = {'model': net.state_dict(), 'opt': opt.state_dict(), 'step': 1, 'seed': 0,
        'rng_torch': torch.get_rng_state(), 'norm_stats': {'mean': torch.zeros(4), 'std': torch.ones(4)}}
torch.save(ckpt, '/tmp/ckpt.pt')
loaded = torch.load('/tmp/ckpt.pt', map_location='cpu', weights_only=False)
net.load_state_dict(loaded['model'], strict=True); opt.load_state_dict(loaded['opt'])
```

`map_location='cpu'` lets a GPU checkpoint load on a machine without one. Recent PyTorch defaults `weights_only=True`, which refuses arbitrary Python objects; pass `False` only for your own files. To resume a run exactly you also need the scheduler's state, the data sampler's epoch and the RNG states; say which ones you saved. Save the normalisation statistics **inside** the checkpoint so the robot cannot use different ones.

*Serves: 21, 22, 24.*

## P2.13 Distributed data parallel (to discuss, not to write)

DDP (distributed data parallel; P1.16) runs one process per GPU (each process has a number, its **rank**; rank 0 is by convention the one that logs and saves), each with a full copy of the model and a different shard of the data. After `backward()`, gradients are averaged across processes with an all-reduce (a collective operation in which every process sends its gradients and receives the average), so every copy takes the same step.

- Launch with `torchrun --nproc_per_node=8 train.py`; in the script, `torch.distributed.init_process_group('nccl')`, read `LOCAL_RANK` from the environment, `torch.cuda.set_device(local_rank)`, wrap `model = DDP(model, device_ids=[local_rank])`.
- **`DistributedSampler`** gives each rank a disjoint shard; call `sampler.set_epoch(epoch)` every epoch or every epoch reuses the same shuffle.
- **Effective batch** = per-GPU batch × number of GPUs × accumulation steps. Going 1 → 8 GPUs multiplies it by 8; the learning rate usually has to change with it (linear scaling with warm-up is the common rule), which is the first thing that "breaks".
- **Gradient accumulation** under DDP: wrap the non-final micro-steps in `with model.no_sync():` so the all-reduce happens only once per optimiser step.
- Only **rank 0** logs, evaluates and saves checkpoints; save `model.module.state_dict()` (the unwrapped model), or keys get a `module.` prefix that later breaks strict loading.
- BatchNorm statistics are per-process unless you convert to `SyncBatchNorm`; LayerNorm models do not care.
- FSDP shards parameters, gradients and optimiser state across ranks instead of replicating them; you need it when the model plus Adam state (about 16 bytes per parameter in mixed precision) does not fit on one GPU.

*Serves: 24, chapter 14.3.*

## P2.14 Debugging tools

The fixed order from the coding list: shapes and dtypes at every boundary → overfit one batch → one hand-computed example through the pipeline → `eval()`, normalisation, `detach` → hyper-parameters. The tools:

```python
t = torch.randn(2, 3)
assert t.shape == (2, 3), t.shape                        # cheapest test there is; put one per function
torch.isfinite(t).all()                                  # NaN/inf check
torch.testing.assert_close(t, t.clone())                 # compare with a reference, with sensible tolerances
grad_norms = {n: p.grad.norm().item() for n, p in net.named_parameters() if p.grad is not None}
sorted(grad_norms.items(), key=lambda kv: -kv[1])[:3]     # which layers get the largest gradients
```

- `torch.autograd.set_detect_anomaly(True)` makes the backward pass report which forward op produced a NaN (slow; debugging only).
- `module.register_forward_hook(fn)` prints or records intermediate shapes and statistics without editing `forward`.
- A layer whose `grad` is `None` after `backward()` is disconnected from the loss (a missing term, an accidental `detach`, a parameter created outside `__init__`). A layer whose grad norm is exactly 0 is behind a dead ReLU, a mask, or a `no_grad`.
- **Collapse check** for any JEPA you write: the per-dimension standard deviation of the embeddings across a batch. If it goes towards 0 while the loss goes down, the model is predicting a constant (chapter 12.2).

*Serves: the debugging drill, 03, 14, 15.*

## P2.15 Primitives for the robot-learning items

These are the building blocks of items 11–19 and 23. The exercises are combining them.

**EMA of parameters** (14; concept in P1.12). Inside `torch.no_grad()`, for each pair of online and target parameters: target ← τ·target + (1−τ)·online, with τ close to 1 (0.99–0.999). `tgt.lerp_(online, 1 − τ)` does exactly this in place. The target's parameters have `requires_grad=False`; it is never given to the optimiser.

```python
online = nn.Linear(3, 3); target = copy.deepcopy(online)
for p in target.parameters(): p.requires_grad_(False)
tau = 0.99
with torch.no_grad():
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.lerp_(po, 1 - tau)
```

**Random unit directions** (15). `u = torch.randn(D, M); u = u / u.norm(dim=0, keepdim=True)` gives M directions on the unit sphere; `z @ u` projects N embeddings onto them, shape (N, M). The SIGReg idea (12c): each column of projections should look like N(0, 1). A moment-matching stand-in penalises the column means away from 0 and variances away from 1; the Epps–Pulley test compares the empirical characteristic function, mean over samples of exp(i·t·x), with exp(−t²/2) at a few t values. `torch.cos` and `torch.sin` give the real and imaginary parts.

**Flow matching** (11, 12; concept in P1.14). x₀ ~ N(0, I) noise, x₁ the data, t ~ U(0, 1) per sample with shape (B, 1, …) so it broadcasts; xₜ = (1 − t)·x₀ + t·x₁; the network predicts v(xₜ, t) and the target is x₁ − x₀. Sampling integrates dx/dt = v from t = 0 to 1 with N Euler steps of size 1/N, starting from a seeded noise tensor (`torch.Generator().manual_seed(s)` and `torch.randn(shape, generator=g)`). Same generator seed, same output: that is the parity habit of chapter 6.

**Gaussian sampling for CEM** (23; concept in P1.13). N candidate sequences of horizon H and action dim A: `mu + std * torch.randn(N, H, A)` with `mu`, `std` of shape (H, A) broadcasting. Clamp to action limits. Cost per candidate is a reduction over the rolled-out latent, shape (N,). Elites: `cost.topk(k, largest=False).indices`. Refit: mean and standard deviation of the elites along dim 0, plus a small floor on std so it does not collapse to zero in one iteration.

```python
N, Hh, A, k = 64, 5, 2, 8
mu, std = torch.zeros(Hh, A), torch.ones(Hh, A)
cand = mu + std * torch.randn(N, Hh, A)                  # (N, H, A)
cost = cand.pow(2).sum(dim=(1, 2))                       # (N,): a stand-in cost
elite = cand[cost.topk(k, largest=False).indices]        # (k, H, A)
mu, std = elite.mean(0), elite.std(0).clamp(min=1e-3)
```

**Autoregressive rollout** (14, 23). A Python loop over the horizon: z ← predictor(z, a[:, h]); keep the list of predicted latents if the cost needs more than the last. Batch the N candidates through the predictor together; a loop over candidates is the slow version an interviewer will ask you to remove.

**Bootstrap over episodes** (17; concept in P1.15). Put per-episode results in a tensor of shape (E,). Resample episode indices with replacement, `idx = torch.randint(0, E, (R, E), generator=g)`, compute the statistic per row, `res[idx].float().mean(1)` (shape (R,)), then take the 2.5 % and 97.5 % quantiles with `torch.quantile`. For a paired difference, resample the **same** episode indices for both arms.

**Wilson interval** (18). For k successes in n trials and z = 1.96: centre (k + z²/2)/(n + z²), half-width z·√(k(n−k)/n + z²/4)/(n + z²). Plain Python with `math.sqrt` is fine.

**AUROC by ranks** (19). The probability that a random positive scores higher than a random negative: rank all scores (`scores.argsort().argsort()` gives 0-based ranks; ties need average ranks for an exact match with scikit-learn), sum the ranks of the positives, subtract n₊(n₊ − 1)/2, divide by n₊·n₋. Or the O(n₊·n₋) version with broadcasting, `(pos[:, None] > neg[None, :]).float().mean()` plus half the ties, which is shorter and fine for interview sizes.

**Chunk executor** (13). Plain Python and tensors, no autograd: a buffer of pending actions, a counter of control steps, a call to the policy every chunk with the delay d simulated as "the chunk arrives d steps after it was requested", and at the swap the first d actions of the new chunk overwritten by what was actually executed (prefix overwrite). Measure the jump as the norm of the difference between the last action of the old chunk and the first executed action of the new one.

*Serves: 11–19, 23.*

## P2.16 Testing your own code in the room

- Use tiny sizes you can check by hand: B = 2, T = 3, D = 4. Different numbers for every axis, so a transposed axis shows up as a shape error instead of passing silently.
- Compare against the PyTorch reference wherever one exists, with `torch.testing.assert_close`: `nn.MultiheadAttention` (02), `F.cross_entropy` (04), `nn.LayerNorm` (07), autograd (08), `torch.optim.Adam` (06), `sklearn.metrics.roc_auc_score` (19).
- For things with no reference, test a property: rows of attention weights sum to 1; the causal mask gives zero weight above the diagonal; the EMA target moves by exactly (1 − τ) times the gap; CEM's elite cost decreases every iteration; the bootstrap interval widens when you halve the number of episodes.
- Say which test you are running and why before you run it. That is half of what is being scored.

## P2.17 Map: exercise → sections → reference to test against

| # | Exercise | Sections | Test against |
|---|---|---|---|
| 01 | Scaled dot-product attention + causal mask | 1, 2, 5, 20 | `F.scaled_dot_product_attention(is_causal=True)` |
| 02 | Multi-head self-attention | 1, 4, 5, 20 | `nn.MultiheadAttention(batch_first=True)` with copied weights |
| 03 | Training loop | 7, 8, 9, 10, 19, 20 | overfit one batch |
| 04 | Cross-entropy by hand | 1, 2, 6 | `F.cross_entropy` |
| 05 | MSE / Huber / L1 with mask | 6 | hand count of valid steps |
| 06 | SGD-momentum, Adam | 3, 7 | `torch.optim.SGD`, `torch.optim.Adam`, one step |
| 07 | LayerNorm, RMSNorm | 1, 2, 5, 20 | `nn.LayerNorm`, `nn.RMSNorm` |
| 08 | Linear with manual backward | 3, 19 | autograd; `gradcheck` in float64 |
| 09 | Tiny MLP end to end | 4, 6, 7, 9 | accuracy on separable data |
| 10 | Warm-up + cosine | 7 | three boundary values |
| 11 | Flow-matching step | 15 | loss decreases on 2-D toy |
| 12 | Euler sampler | 15, 10 | same seed → same output |
| 13 | Chunk executor + prefix overwrite | 15 | jump smaller with overwrite |
| 14 | JEPA step | 3, 4, 15, 14 | target `.grad is None`; EMA moves; per-dim std not → 0 |
| 15 | SIGReg | 1, 2, 15 | constant embeddings → large penalty |
| 16 | Baselines behind one `evaluate()` | 4 | one call works for three models |
| 17 | Episode bootstrap CI | 15 | widens with fewer episodes |
| 18 | Wilson | 15 | 7/10 → (0.40, 0.89) |
| 19 | AUROC by hand | 1, 15 | `sklearn.metrics.roc_auc_score` |
| 20 | Episodic dataset, padding mask, episode split | 8, 6 | disjoint ids; mask zero on padding |
| 21 | Train-only normalisation stats | 8, 12 | val split change leaves stats unchanged |
| 22 | Strict load with count | 4, 12 | missing key raises |
| 23 | Latent CEM / MPC | 1, 15 | toy linear predictor reaches the goal |
| 24 | DDP skeleton (whiteboard) | 10, 11, 12, 13 | the 1 → 8 GPU list |

## P2.18 Exercises for this chapter

1. Run every code block in this chapter in a REPL, in order. For each, change one thing (an axis, a dtype, `keepdim`, the mask convention) and predict the result before pressing enter. Write down every prediction you got wrong; those are the sections to re-read.
2. Without looking, write the shape after every line of the split-heads / merge-heads block in P2.1, then check.
3. Derive the three linear-layer gradients of P2.3 on paper from the index form, then confirm with `gradcheck`.
4. Explain, in two sentences each, out loud: why `detach` on the JEPA target; why `eval()` and `no_grad()` are different things; why bf16 needs no loss scaler; why the mask convention differs between `nn.MultiheadAttention` and `F.scaled_dot_product_attention`; what breaks going from 1 to 8 GPUs.

## P2.19 Worked build 1: an autograd engine from scratch

Everything in P2.3 happens inside PyTorch where you cannot see it. This build makes it visible: a scalar `Value` that records how it was computed and can backpropagate, in about sixty lines of plain Python. Once you have written this yourself, `loss.backward()`, gradient accumulation and `zero_grad` stop being magic; it is also exactly the depth an interviewer reaches with "how does autograd work?". Read it once, run it, then close the book and write it again from an empty file (item 08 is the tensor version).

**The idea.** Each `Value` stores its number (`data`), its gradient (`grad`, the derivative of the final output with respect to it, filled in by `backward`), the `Value`s it was computed from (`_prev`), and a small function `_backward` that knows the *local* derivative of the operation that created it. `backward()` on the final output sets its own gradient to 1, sorts the graph so that every node comes after its inputs (a **topological order**), and walks it in reverse, letting each node push its gradient to its inputs with the chain rule (P1.3).

```python
import math, random

class Value:
    """A scalar that remembers how it was computed, so it can backpropagate."""
    def __init__(self, data, _children=(), _op=''):
        self.data = data            # the number
        self.grad = 0.0             # d(final output)/d(this), filled by backward()
        self._backward = lambda: None   # how to push my grad to my inputs
        self._prev = set(_children)     # the Values I was computed from
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad          # d(a+b)/da = 1
            other.grad += 1.0 * out.grad         # d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):                        # k is a plain number
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad  # d tanh(x)/dx = 1 - tanh(x)^2
        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()
        def build(v):                            # topological order: inputs before outputs
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0                          # d(out)/d(out)
        for v in reversed(order):                # outputs first, then their inputs
            v._backward()

    # conveniences so expressions read naturally
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
```

Four details carry most of the understanding:

- **`+=`, not `=`, in every `_backward`.** A value used twice (say `a * a`, or a weight shared by two neurons) receives gradient from both uses, and the contributions add. This is the same "gradients accumulate" of P2.3, and the reason `zero_grad` exists.
- **Each operation only knows its local derivative.** Addition passes the gradient through unchanged; multiplication passes it scaled by the *other* input; tanh scales it by 1 − tanh². The chain rule is the multiplication by `out.grad`.
- **The order matters.** A node may push its gradient only after it has received gradient from every node that uses it; the reversed topological order guarantees this.
- **No tensors, no GPU.** PyTorch does the same with tensors (whole arrays per node, so far fewer nodes) and with a precompiled `_backward` per operation.

Check it on the example of P1.3:

```python
# the chapter P1.3 example: L = (w1*x1 + w2*x2 - 1)^2 with w = (2, -1), x = (3, 4)
w1, w2 = Value(2.0), Value(-1.0)
L = (w1 * 3 + w2 * 4 - 1) ** 2
L.backward()
print(L.data, w1.grad, w2.grad)       # 1.0 6.0 8.0
```

**A neural network on top.** A neuron is a weighted sum plus a bias through tanh; a layer is a list of neurons; an MLP is a list of layers. Train it on four points with the loop of P2.9 written by hand: forward, loss, zero the gradients, backward, step.

```python
class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)
    def __call__(self, x):
        return sum((wi * xi for wi, xi in zip(self.w, x)), self.b).tanh()
    def parameters(self):
        return self.w + [self.b]

class MLP:
    def __init__(self, sizes):                   # e.g. [3, 4, 4, 1]
        self.layers = [[Neuron(sizes[i]) for _ in range(sizes[i + 1])] for i in range(len(sizes) - 1)]
    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]

random.seed(0)
net = MLP([3, 4, 4, 1])
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
for step in range(50):
    preds = [net(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))
    for p in net.parameters():
        p.grad = 0.0                             # zero_grad
    loss.backward()
    for p in net.parameters():
        p.data -= 0.05 * p.grad                  # SGD step
    if step % 10 == 0 or step == 49:
        print(step, round(loss.data, 4))
print([round(net(x).data, 2) for x in xs])
print(len(net.parameters()))
```

On the run of 2026-10-24 the loss fell 3.59 → 0.126 (step 10) → 0.016 (step 49) and the predictions reached (0.94, −0.94, −0.93, 0.93) for targets (1, −1, −1, 1); 41 parameters. Try: remove the `p.grad = 0.0` line and watch training fail (gradients accumulate across steps); set the learning rate to 1.0 and watch it diverge (P1.2).

*Serves: 03, 06, 08, and the "how does autograd work" question.*

## P2.20 Worked build 2: a tiny GPT, end to end

This build assembles every piece of P1.10 and P2.1–P2.9 into one working model: a character-level transformer that learns to continue a short text. It is the reference for items 01, 02, 03 and 07, and it is the same architecture as the predictors and language backbones of the world-model papers, at toy scale. It trains on a laptop CPU in about twenty seconds.

**The task.** Language modelling: given a sequence of characters, predict the next character at every position. The text is turned into integers (one id per distinct character), and a training example is a window of T characters with the same window shifted by one as the target.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# ---- data: characters of a short text, as integers ----
text = ("the arm moves the red brick onto the plate. the arm returns to rest. "
        "the judge weighs the plate. the policy predicts a chunk of actions. ") * 40
chars = sorted(set(text))
V = len(chars)                                   # vocabulary size
stoi = {c: i for i, c in enumerate(chars)}       # char -> id
itos = {i: c for c, i in stoi.items()}           # id -> char
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train, val = data[:n], data[n:]

T, B = 32, 16                                    # context length, batch size
def get_batch(split):
    d = train if split == 'train' else val
    ix = torch.randint(0, len(d) - T - 1, (B,))  # B random start positions
    x = torch.stack([d[i:i + T] for i in ix])            # (B, T) inputs
    y = torch.stack([d[i + 1:i + T + 1] for i in ix])    # (B, T) targets: the next char
    return x, y
```

Note what `get_batch` returns: `x` and `y` are both (B, T), and `y[b, t]` is the character that follows `x[b, t]`. One window gives T training signals at once, which is why the attention must be causal: position t must not see position t + 1, its own answer.

**The layers.** One attention head (P1.10 in code: queries, keys and values by three linear layers; scores scaled by √d; the causal mask as a lower-triangular boolean buffer where True means "may attend"; softmax; weighted sum of values); several heads concatenated and projected; a position-wise MLP that widens to 4·D and back; and the block that wraps both in LayerNorm and a residual connection.

```python
D, H, L, p_drop = 64, 4, 2, 0.1                  # width, heads, blocks, dropout

class Head(nn.Module):
    """One causal self-attention head."""
    def __init__(self, Dh):
        super().__init__()
        self.q = nn.Linear(D, Dh, bias=False)
        self.k = nn.Linear(D, Dh, bias=False)
        self.v = nn.Linear(D, Dh, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(T, T, dtype=torch.bool)))  # True = may attend
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):                        # x: (B, t, D)
        b, t, _ = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)                    # each (B, t, Dh)
        att = q @ k.transpose(-2, -1) / k.shape[-1] ** 0.5           # (B, t, t) scores
        att = att.masked_fill(~self.mask[:t, :t], float('-inf'))     # hide the future
        att = self.drop(F.softmax(att, dim=-1))                      # rows sum to 1
        return att @ v                                               # (B, t, Dh)

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head(D // H) for _ in range(H)])
        self.proj = nn.Linear(D, D)
        self.drop = nn.Dropout(p_drop)
    def forward(self, x):
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))  # (B, t, D)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(p_drop))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Pre-norm transformer block: x + attention(norm(x)), then x + mlp(norm(x))."""
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attn, self.mlp = MultiHead(), FeedForward()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))           # residual around attention
        x = x + self.mlp(self.ln2(x))            # residual around the MLP
        return x
```

Details worth saying out loud in an interview: the mask is a **buffer** (saved and moved with the model, never trained); it is sliced to `[:t, :t]` so that shorter inputs work at generation time; dropout is applied to the attention weights and after each projection; the block is **pre-norm** (normalise, then attend, then add), the arrangement modern transformers use because it trains more stably than normalising after the residual.

**The model.** Token embeddings plus position embeddings (P1.10: attention alone does not know order), the blocks, a final LayerNorm, and a linear **head** to V logits per position. The loss is the cross-entropy of P2.6 with the (B, T, V) logits flattened to (B·T, V). Generation samples one character at a time from the softmax of the last position and appends it.

```python
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)            # id -> vector
        self.pos = nn.Embedding(T, D)            # position -> vector
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)              # vector -> logits over the vocabulary
    def forward(self, idx, targets=None):        # idx: (B, t) ids
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))  # (B, t, D) + (t, D)
        logits = self.head(self.ln(self.blocks(x)))                       # (B, t, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(b * t, V), targets.view(b * t))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx[:, -T:])        # crop to the context length
            probs = F.softmax(logits[:, -1, :], dim=-1)          # last position only: (B, V)
            nxt = torch.multinomial(probs, num_samples=1)        # sample one id
            idx = torch.cat([idx, nxt], dim=1)
        return idx
```

**Training.** The loop of P2.9: AdamW, zero_grad, backward, step, and an evaluation function that switches to `eval()` and `no_grad()` and back.

```python
model = TinyGPT()
print(sum(p.numel() for p in model.parameters()), 'parameters')
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

@torch.no_grad()
def estimate(split, iters=20):
    model.eval()
    losses = [model(*get_batch(split))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)

print('start', round(estimate('train'), 3), 'ln V =', round(torch.log(torch.tensor(float(V))).item(), 3))
for step in range(301):
    x, y = get_batch('train')
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(step, 'train', round(estimate('train'), 3), 'val', round(estimate('val'), 3))

model.eval()
out = model.generate(torch.tensor([[stoi['t']]]), 80)
print(''.join(itos[i] for i in out[0].tolist()))
```

On the run of 2026-10-24: 104 985 parameters; loss at initialisation 3.42, against ln V = 3.22; train/val 3.07 → 0.19 → 0.09 → 0.09 over 300 steps; the sample reproduced the training sentences ("the policy predicts a chunk of actions. the arm moves the red brick onto the plat…"). Three things to learn from those numbers:

- **The loss at initialisation should be about ln V.** A random model spreads probability evenly over V characters, and −ln(1/V) = ln V. If your first loss is far above it, the initialisation or the loss is wrong. This is the cheapest sanity check in deep learning; say it in the debugging round.
- **Train and validation loss move together here** because the text repeats; on real data a gap opening between them is overfitting (P1.7).
- **The model memorised the text.** With a 160-character text repeated, that is all there is to learn. Swap in any longer text file to see it generalise imperfectly.

Things to try, each a one-line change; write your prediction before running. Two were run on 2026-10-24: **removing the causal mask** drove the loss to 0.004 (the model reads its own answers through attention) and generation degenerated to "tttttttt…", because at generation time the answers are not there to read; **removing the position embeddings** barely changed the loss on this repetitive text (0.089 vs 0.088), because a causal model can partly infer position from how much context it has seen, and the sample became slightly less coherent — a surprise worth remembering when someone says attention "cannot" know order. Not run here, predict and check yourself: removing the residual connections with L = 6; setting dropout to 0 on a longer, non-repeating text.

**From this to the coding list.** Item 01 is `Head.forward` with the scaled dot-product written as its own function. Item 02 is `MultiHead` written as one module with a single (D → 3D) projection and the split-heads reshaping of P2.1 instead of a list of heads, checked against `nn.MultiheadAttention`. Item 03 is the training section. Item 07 is LayerNorm written by hand in place of `nn.LayerNorm`. Item 14 (the JEPA step) reuses the same blocks as its predictor, with an action embedding added to the input and an MSE loss on latents instead of cross-entropy on characters.

*Serves: 01, 02, 03, 07, 09, 14, and the "explain a transformer" question.*
