# Week 1: PyTorch from zero

**Everything you need for this week is on this page.** You do not need to open `knowledge/`,
`reference/` or any other folder to do the five labs. Links to them at the end are optional depth.

You need: Python, high-school maths, and `uv sync` done once at the repo root. Nothing else.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [Deep learning in ten minutes](#deep-learning-in-ten-minutes) (once, Monday, 10 min).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-tensors-and-shapes),
   [Day 2](#day-2-indexing-broadcasting-einsum), [Day 3](#day-3-autograd),
   [Day 4](#day-4-nnmodule-and-the-training-loop), [Day 5](#day-5-behavior-cloning-on-a-toy-so-101).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

Every day folder has the same seven files. Only two of them are yours to edit.

| File | What it is | You |
|---|---|---|
| `predict.md` | 5–10 small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day (the short version of this page) | read |
| `predict_run.py` | prints the real answers to `predict.md` | run, compare |
| `w1dN_*.py` | **the exercise**: functions with `# TODO` | **write the code** |
| `test_w1dN_*.py` | tests for the exercise | run, never edit |
| `review_w1dN.py` + `test_review_w1dN.py` | code that looks right and has one bug | **find and fix the bug** |
| `questions.md` | interview questions on the day | answer out loud |

### What to do each day

One hour. Run every command from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve old cards (nothing on day 1) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` in the day folder |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w1-pytorch-basics/<day>/predict_run.py` |
| 20–45 | Build | fill the TODOs, then `uv run pytest weeks/w1-pytorch-basics/<day> -k "not review"` until green |
| 45–53 | Review | fix the bug, then `uv run pytest weeks/w1-pytorch-basics/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w1-pytorch-basics/<day>` |

Guess before reading even if you know nothing: a wrong guess makes the lesson stick better
(`LEARNING.md` rule 1). AI stays off in the exercise files; stuck for 10 minutes, ask for a hint, not code.

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-tensors-shapes` | tensors, shapes, reshaping, reductions | padding with masks, per-joint normalisation, action chunks | a mean over the wrong axis |
| Tue | `d2-indexing-broadcasting` | broadcasting, `gather`, masks, einsum | pairwise distances, masked mean, best-of-N | a masked loss divided by the wrong count |
| Wed | `d3-autograd` | gradients, the chain rule, `backward`, `detach` | linear-layer gradients by hand, gradient descent | gradients that are never zeroed |
| Thu | `d4-modules-training-loop` | `nn.Module`, optimiser, train and eval modes | an MLP, the training loop, overfit one batch | validation with dropout on |
| Fri | `d5-bc-so101-toy` | behavior cloning, episode splits, normalisation stats, checkpoints | a policy that beats "hold position" on a toy SO-101 | a split that leaks episodes |

By Friday you can write a complete training script for a small robot policy from an empty file,
and say what each line protects against. Weeks 2–9 all build on that.

## Deep learning in ten minutes

Read this once. Every later section uses these words.

**Model, parameters.** A model is a function with adjustable numbers inside. The numbers are the
**parameters** (or weights). The smallest model: `ŷ = w · x + b`, two parameters.

**Loss.** One number that says how wrong the model is on some examples. The usual one for
numbers is the **mean squared error (MSE)**: average of `(ŷ − y)²`.

Three examples `(x, y)`: (1, 3), (2, 5), (3, 7). The true rule is `y = 2x + 1`. With `w = 0, b = 0`
the model says 0 each time; the errors are −3, −5, −7; squared 9, 25, 49; the loss is **27.67**.

**Gradient.** For each parameter, the slope of the loss: "if I increase this parameter a little,
how much does the loss change?". Here `∂L/∂w = mean(2·(ŷ − y)·x) = −22.67` and
`∂L/∂b = mean(2·(ŷ − y)) = −10`. Both negative: increasing `w` or `b` lowers the loss.

**Gradient descent.** Step each parameter against its gradient: `w ← w − η · ∂L/∂w`. `η` is the
**learning rate**, the step size. With `η = 0.05`: after one step `w = 1.13, b = 0.5`, loss 5.49;
after 500 steps `w = 2, b = 1`, loss 0. With `η = 0.3` every step overshoots and the loss grows:
27.7 → 150 → 812. "Loss goes up, then NaN" usually means the learning rate is too large.

**Backpropagation.** A real model has millions of parameters; nobody derives the gradients by hand.
The computer records every operation of the forward computation (the **computation graph**) and
applies the chain rule backwards through it. In PyTorch that is one call, `loss.backward()` (day 3).

**Training** is this loop and nothing more: forward → loss → gradients → small step → repeat.

| Word | Meaning |
|---|---|
| batch | several examples stacked in one tensor and processed together |
| step | one batch → loss → gradients → update |
| epoch | one pass over all the training data |
| train / validation split | data the model learns from / data you only measure on |
| overfitting | training loss falls, validation loss rises: the model memorises |
| hyper-parameter | a setting you choose (learning rate, width); a parameter is learned |

All code on this page assumes:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

Open a Python REPL (`uv run python`) and paste the snippets. Change one thing, predict the
output, press enter. The snippets teach the tools; none of them is an exercise answer.

---

## Day 1: Tensors and shapes

Folder: `d1-tensors-shapes`. Exercise: `w1d1_tensors.py`.

### Why this matters

Most bugs in robot-learning code are shape bugs that **do not crash**: the code runs and the
numbers are wrong. The habit that prevents them: say the shape of every tensor, and write it in a
comment on every line.

### A tensor is four things

A block of numbers, plus a **shape** (size of each axis), a **dtype** (number format) and a
**device** (CPU or GPU).

```python
x = torch.zeros(4, 50, 6)        # (B, T, D): 4 episodes, 50 steps, 6 joints
x.shape, x.dtype, x.device       # torch.Size([4, 50, 6]), torch.float32, cpu
x.ndim, x.numel()                # 3 axes, 1200 numbers
```

Robot data uses the same few letters everywhere. Learn to read them:

| Letter | Meaning | SO-101 example |
|---|---|---|
| `B` | batch: independent samples | 32 |
| `T` | time steps in an episode or a window | 50 |
| `H` | horizon of an action chunk | 50 for ACT and SmolVLA |
| `D` | features per step | 6 (5 joints + gripper) |
| `C, Hh, W` | image channels, height, width | 3, 224, 224 |

### Creating tensors, and the dtype trap

```python
torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # from Python lists
torch.zeros(2, 3), torch.ones(2, 3), torch.full((2, 3), 7.0)
torch.arange(5)                                  # 0, 1, 2, 3, 4
torch.randn(2, 3)                                # random, mean 0, std 1
torch.zeros_like(x)                              # same shape, dtype, device as x
```

The dtype is inferred from the data. Python integers give an integer tensor, and integer tensors
cannot hold gradients or fractional values. Robot data arrives as lists and arrays of mixed
types, so state the dtype yourself: `torch.tensor(data, dtype=torch.float32)`.

| dtype | Use |
|---|---|
| `float32` | the default for data and weights |
| `int64` (`torch.long`) | indices: class labels, positions for `gather` |
| `bool` | masks |
| `float16`, `bfloat16` | half precision, to train faster on a GPU; about 3 digits of precision |
| `float64` | checking gradients only |

### Indexing

Same rules as NumPy.

```python
x = torch.randn(4, 50, 6)        # (B, T, D)
x[0]                             # (T, D): first episode
x[:, -1]                         # (B, D): last step of every episode
x[..., 5]                        # (B, T): the gripper everywhere
x[:, None].shape                 # (B, 1, T, D): None inserts an axis
```

### Changing the shape

Four families. They do different things; mixing them up is the classic bug.

| Operation | What it does | Copies memory? |
|---|---|---|
| `view(shape)` | same memory, read with a new shape | never; **fails** if the memory is not contiguous |
| `reshape(shape)` | like `view`, copies when it has to | sometimes |
| `transpose(a, b)`, `permute(dims)` | reorder the axes | never; the result is **not contiguous** |
| `unsqueeze(d)`, `squeeze(d)` | add or remove an axis of size 1 | never |

"Contiguous" means the numbers sit in memory in the order the shape reads them. `transpose`
does not move any number: it only changes how the tensor walks its memory. After it, `view` cannot
work and raises an error; `reshape` (or `.contiguous().view(...)`) can.

```python
img = torch.randn(8, 3, 32, 32)          # (B, C, Hh, W)
flat = img.reshape(8, -1)                # (B, 3072): -1 means "work this one out"
hwc = img.permute(0, 2, 3, 1)            # (B, Hh, W, C): channels last
hwc.is_contiguous()                      # False
hwc.reshape(8, -1).shape                 # works; hwc.view(8, -1) raises
```

**Reshape and permute are not the same thing.** `reshape` keeps the order of the numbers and cuts
them differently. `permute` changes which axis is which. To turn `(B, C, Hh, W)` into
`(B, Hh, W, C)` you need `permute`. A `reshape` to that shape runs, and scrambles the image.

### Combining tensors

```python
a, b = torch.zeros(6), torch.ones(6)
torch.stack([a, b], dim=0).shape     # (2, 6): a NEW axis
torch.cat([a, b], dim=0).shape       # (12,): along an EXISTING axis
```

Stack makes a batch out of samples. Cat makes a longer sequence out of sequences.

### Reductions: `dim=k` removes axis k

```python
x = torch.randn(4, 50, 6)                     # (B, T, D)
x.mean(dim=0).shape                           # (T, D): averaged over episodes
x.mean(dim=(0, 1)).shape                      # (D,): one number per joint
x.mean(dim=-1, keepdim=True).shape            # (B, T, 1): axis kept with size 1
```

`sum`, `mean`, `std`, `max`, `min` all take `dim=` and `keepdim=`. `keepdim=True` is what you
want when the result goes back into arithmetic with the original tensor (day 2).

"Per-joint statistics" means: remove every axis except the joint axis. Ask yourself which axis
must **survive**, then reduce the others. Getting this wrong still returns a tensor, sometimes
of the right shape when two axes happen to have the same size. That is today's review bug.

### Padding and masks

Episodes have different lengths, and a tensor is a rectangle. So you allocate the rectangle for
the longest episode, copy each episode in, and keep a boolean **mask** that says which cells are real.

```python
lengths = [3, 5, 2]
grid = torch.zeros(len(lengths), max(lengths))        # (N, T_max), zeros = padding
mask = torch.zeros(len(lengths), max(lengths), dtype=torch.bool)
grid[0, :3] = 1.0                                      # copy one row in...
mask[0, :3] = True                                     # ...and mark it real
```

The mask travels with the data from now on. Every loss and every mean must use it, or the padding
is counted as real data (day 2).

### Normalisation

Shoulder angles span about −150 to 150 degrees; the gripper spans 0 to 100. A network fed raw
values listens mostly to the largest numbers. So each feature is rescaled to mean 0 and standard
deviation 1: `z = (x − mean) / std`, with one mean and one std **per joint**.

One trap: a joint that never moves in the data has `std = 0`, and dividing by zero gives `inf` or
`NaN`. A small `eps` in the denominator prevents the crash. (It also turns tiny sensor noise on
that joint into big numbers; `questions.md` asks you for the better fix.)

### Sliding windows with `unfold`

A chunked policy (ACT, SmolVLA) predicts the next `H` actions at once. Its training targets are
all windows of `H` consecutive actions. `unfold` produces sliding windows without copying anything.

```python
t = torch.arange(10)                 # (10,)
w = t.unfold(0, 4, 1)                # dimension 0, window size 4, step 1
w.shape, w[0], w[1]                  # (7, 4), tensor([0, 1, 2, 3]), tensor([1, 2, 3, 4])
```

`T − size + 1` windows. On a tensor with more axes, `unfold` adds the window axis **at the end**.
Print the shape, and compare `w[0]` with the slice you expect, before you trust it.

### Devices

```python
torch.backends.mps.is_available()    # Apple GPU
torch.cuda.is_available()            # NVIDIA GPU
x.to("cpu")                          # move a tensor; model.to(device) moves a model
```

Every tensor in one operation must be on the same device. Pick the device once at the top of a script.

### Traps of the day

- `view` after `transpose` or `permute` raises. Use `reshape`.
- Integers in the input list give an integer tensor. State `dtype=torch.float32`.
- A reduction over the wrong axis returns a tensor of a plausible shape. Check on a tensor whose
  axes all have different sizes (2, 3, 4), never (6, 6).
- `reshape` where `permute` was needed: right shape, scrambled content.

### Today's build

`w1d1_tensors.py`. What each function is for, and the section that covers it:

| Function | Job | Lesson section |
|---|---|---|
| `to_tensor` | Python lists → float32 tensor | the dtype trap |
| `flatten_time`, `unflatten_time` | `(B, T, D) ↔ (B·T, D)`; every per-frame network needs it | changing the shape |
| `pad_episodes` | list of `(T_i, D)` → `(N, T_max, D)` plus a `(N, T_max)` mask | padding and masks |
| `normalize` | per-joint z-score with `eps` | normalisation |
| `action_chunks` | `(T, D)` → `(T−H+1, H, D)` | sliding windows |
| `pick_device` | mps, else cuda, else cpu | devices |

Run: `uv run pytest weeks/w1-pytorch-basics/d1-tensors-shapes -k "not review"`

---

## Day 2: Indexing, broadcasting, einsum

Folder: `d2-indexing-broadcasting`. Exercise: `w1d2_broadcast.py`.

### Why this matters

**No Python loop over the batch.** A loop is about 100 times slower on a GPU, and interviewers
watch for it. Four tools replace loops: broadcasting, index selection, masks and einsum.

### Broadcasting

When two tensors of different shapes meet in `+ − * /`, PyTorch lines their shapes up **from the
right**. Each pair of sizes must be equal, or one of them must be 1. An axis of size 1 is
repeated to match. A missing axis on the left counts as size 1.

```
(4, 50, 6)        (4, 50, 6)        (4, 50, 6)
      (6,)        (4,  1, 1)           (4, 6)
----------        ----------        ---------
(4, 50, 6) ok     (4, 50, 6) ok     error: 50 meets 4
```

```python
x = torch.randn(4, 50, 6)                    # (B, T, D)
x - x.mean(dim=(0, 1))                       # (B, T, D) − (D,): one mean per joint
x / x.amax(dim=-1, keepdim=True)             # (B, T, D) / (B, T, 1): this is why keepdim exists
```

The trick that removes a double loop: put the two things on **different axes**, each with a 1
where the other has its size, and let broadcasting build the grid.

```python
r = torch.arange(3)[:, None]        # (3, 1)
c = torch.arange(4)[None, :]        # (1, 4)
(r * 10 + c).shape                  # (3, 4): every r against every c
```

"Every row of `a` against every row of `b`" is the same idea with one more axis. The price is
memory: the intermediate result holds every pair.

**The silent failure.** Broadcasting never warns you. A loss between `pred` of shape `(N, 1)` and
`target` of shape `(N,)` broadcasts to `(N, N)`: every prediction against every target. The code
runs, the loss is a number, and the model learns the mean. Check the two shapes before any loss.

### Selecting with indices

```python
x = torch.arange(12).reshape(3, 4)   # (3, 4)
x[1, 2]                              # one element
x[:, 1]                              # (3,): a column
x[x > 5]                             # 1-D tensor of the elements that pass
x[torch.tensor([0, 2])]              # (2, 4): rows 0 and 2
```

Two index tensors together are paired **element by element**, not as a grid:
`x[rows, cols]` returns `x[rows[0], cols[0]]`, `x[rows[1], cols[1]]`, …

**One element per row** is so common it has its own function. Given class scores `(N, C)` and
the true class of each row `(N,)`, pick each row's score for its true class:

```python
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 3.0]])   # (N=2, C=3)
target = torch.tensor([0, 2])                                # (N,) int64
logits.gather(1, target[:, None]).squeeze(1)                 # (N,): tensor([2., 3.])
```

The rule of `gather(dim, index)`: `index` has the same number of axes as the input, and the
output has the shape of `index`. Hence the `[:, None]` going in and the `squeeze` coming out.
Cross-entropy and Q-learning both use it.

`argmax(dim)` returns the position of the largest value along an axis, `topk(k, dim)` the `k`
largest values and their positions. With positions in hand, you select with the tools above.

### Masks and the masked mean

A boolean mask zeroes or selects. The one rule: **divide by the number of valid items, not by
the total.**

```python
scores = torch.tensor([[1.0, 3.0, 0.0], [2.0, 0.0, 0.0]])        # (B, T), zeros are padding
valid = torch.tensor([[True, True, False], [True, False, False]])
(scores * valid).sum() / valid.sum()         # 2.0: mean of the three real values
scores.mean()                                # 1.0: padding diluted it
```

To multiply a `(B, T, D)` tensor by a `(B, T)` mask, give the mask a last axis: `mask[..., None]`
is `(B, T, 1)` and broadcasts over `D`. Guard the division for a row with no valid item:
`.clamp(min=1)` on the count.

Why it matters on a robot: short episodes have the most padding, so the wrong denominator
down-weights them, and the end of an episode is where the grasp and the place happen.

### einsum

Name each axis with a letter. A letter that is missing from the output is summed over.

```python
A = torch.randn(3, 4); B = torch.randn(4, 5); v = torch.randn(4)
torch.einsum("ij,jk->ik", A, B)      # matrix product: j disappears, so it is summed
torch.einsum("ij,j->i", A, v)        # matrix times vector
torch.einsum("ij->j", A)             # column sums
```

A batch letter that appears on both sides is kept, not summed: that is how one einsum does `B`
independent products at once. Attention scores in week 3 are a single einsum.

`@` is the matrix product. With more than two axes, the leading axes are batch axes and the last
two are multiplied.

### In place, or a new tensor

```python
a = torch.zeros(3); b = a
b += 1          # in place: a changed too, they share memory
b = b + 1       # new tensor: a does not change
```

In-place operations (`+=`, anything ending in `_`) are dangerous on tensors that autograd needs
for the backward pass. You meet the error tomorrow.

### Traps of the day

- Shapes `(N, 1)` and `(N,)` in a loss: silent `(N, N)`.
- Masked mean divided by `T` or by `numel()`.
- `x[rows, cols]` pairs the indices; it does not select a block.
- The broadcast grid can be huge: `(N, M, D)` floats.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `pairwise_sq_dist` | `(N, D)`, `(M, D)` → `(N, M)` squared distances, no loop | broadcasting, the grid trick |
| `select_per_row` | one entry per row | `gather` |
| `masked_mean` | mean over valid time steps only | masks |
| `batched_matvec` | a batch of matrix × vector | einsum |
| `best_of_n` | among N candidate chunks per sample, return the best-scored one | `argmax`, then selecting |

Run: `uv run pytest weeks/w1-pytorch-basics/d2-indexing-broadcasting -k "not review"`

---

## Day 3: Autograd

Folder: `d3-autograd`. Exercise: `w1d3_autograd.py`.

### Why this matters

Training is "move each weight a little in the direction that lowers the loss". Autograd computes
that direction. You must also be able to compute it by hand for a linear layer: interviews ask
for it, and it is how you check that autograd computes what you think.

### The chain rule on one example

`L = (w₁x₁ + w₂x₂ − 1)²` with `w = (2, −1)` and `x = (3, 4)`. Forward, step by step:

1. `s = w₁x₁ + w₂x₂ = 6 − 4 = 2`
2. `e = s − 1 = 1`
3. `L = e² = 1`

Backward, from `L`, multiplying local slopes:

- `∂L/∂e = 2e = 2`
- `∂e/∂s = 1`, so `∂L/∂s = 2`
- `∂s/∂w₁ = x₁ = 3`, so `∂L/∂w₁ = 2 · 3 = 6`; and `∂L/∂w₂ = 2 · 4 = 8`

The same thing in PyTorch:

```python
w = torch.tensor([2.0, -1.0], requires_grad=True)    # "track gradients for this one"
x = torch.tensor([3.0, 4.0])                         # data: no gradient needed
L = ((w * x).sum() - 1.0) ** 2                       # the forward pass builds the graph
L.backward()                                         # the backward pass fills w.grad
w.grad                                               # tensor([6., 8.])
```

### Five facts that cover most questions

1. **`backward()` needs a scalar.** A loss of shape `(B,)` means a `.mean()` is missing.
2. **Gradients accumulate.** A second `backward()` **adds** to `.grad`. So every training step
   clears the gradients first. Forgetting it is today's review bug.
3. **`detach()` cuts the graph.** `y.detach()` has the same values and no history: no gradient
   flows through it. It is the "stop-gradient" of JEPA (week 7).
4. **`torch.no_grad()`** stops recording inside a block. Use it for weight updates and for
   evaluation. It does not change dropout; that is `model.eval()` (day 4).
5. **The graph is freed after `backward()`.** Calling it twice on the same graph raises.

Only **leaves** keep `.grad`: tensors you created yourself with `requires_grad=True`. A tensor
computed from others is not a leaf and its `.grad` is `None`. Only floating-point tensors can
require gradients.

### Gradient descent by hand

The weight update must not itself be recorded, so it goes inside `no_grad`. This is the loop of
"Deep learning in ten minutes", written with autograd, on a different function:

```python
p = torch.tensor(5.0, requires_grad=True)
for _ in range(50):
    loss = (p - 2.0) ** 2            # forward
    loss.backward()                  # p.grad = 2 (p - 2)
    with torch.no_grad():
        p -= 0.1 * p.grad            # step, not recorded
    p.grad.zero_()                   # clear, or the next backward adds to it
p                                    # close to 2.0
```

Delete the `zero_()` line and run it again: `p` swings between about −1 and 5 and never
settles at 2, because each step uses the sum of all past gradients.

### A linear layer, by hand

```
pred = x @ W.T + b          x: (N, I)   W: (O, I)   b: (O,)   pred: (N, O)
L    = mean((pred − y)²)    mean over all N·O numbers
```

Step back from the loss, as in the scalar example:

```
g  = ∂L/∂pred = 2 (pred − y) / (N·O)        (N, O)
dW = gᵀ @ x                                  (O, I)
db = g.sum(dim=0)                            (O,)
dx = g @ W                                   (N, I)
```

The rule that stops you from getting it wrong: **a gradient has the shape of the thing it is
the gradient of.** `dW` must be `(O, I)`. With `g` of shape `(N, O)` and `x` of shape `(N, I)`,
only one product gives `(O, I)`. `b` is added to all `N` rows, so it collects the gradient of all
`N` rows: a sum over the batch axis.

### Finite differences: the universal check

The definition of a slope, applied numerically, one coordinate at a time:

```
∂f/∂xᵢ ≈ ( f(x + ε·eᵢ) − f(x − ε·eᵢ) ) / 2ε        eᵢ = 1 at position i, 0 elsewhere
```

It needs two evaluations of `f` per coordinate, so it is far too slow for training. But it is
always right, which makes it the test for any gradient you wrote by hand. Use `float64` and
`ε ≈ 1e-6`: in `float32` the subtraction of two nearly equal numbers loses most of its digits.
The two-sided ("central") form above is much more accurate than `(f(x + ε) − f(x)) / ε`.

### Stop-gradient

```python
a = torch.tensor(2.0, requires_grad=True)
c = (a * 3).detach() * a       # the first factor is now a constant, 6
c.backward()
a.grad                         # tensor(6.), not 12
```

In a JEPA or BYOL loss, `pred` must move toward `target`, and `target` must not move toward
`pred`. Without the detach, the cheapest way to make them equal is for both to become the same
constant: **collapse**. You build that loss today in its smallest form.

### In-place operations and autograd

Autograd saves some tensors during the forward pass to use them in the backward pass. Changing
one in place (`x += 1`) makes those saved values wrong, and PyTorch raises. Rule: nothing in
place on tensors that are part of the graph; in place is fine inside `no_grad`.

### Traps of the day

- No `zero_()`: gradients add up across steps.
- A weight update outside `no_grad`: error, or an update recorded in the graph.
- `.grad` read on a non-leaf: `None`.
- Gradient check in float32: fails for no real reason.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `linear_mse_grads` | `dW`, `db` by hand; the test compares with autograd | a linear layer, by hand |
| `finite_diff_grad` | central differences for any scalar function | finite differences |
| `fit_line` | gradient descent with autograd, no optimiser class | gradient descent by hand |
| `stop_gradient_mse` | an MSE where only `pred` receives a gradient | stop-gradient |

Run: `uv run pytest weeks/w1-pytorch-basics/d3-autograd -k "not review"`

---

## Day 4: nn.Module and the training loop

Folder: `d4-modules-training-loop`. Exercise: `w1d4_training.py`.

### Why this matters

Every training script is the same few lines. Yesterday you wrote them with raw tensors. Today
PyTorch's own tools carry the parameters (`nn.Module`) and do the update (an optimiser), and you
learn what each line protects against.

### Layers

**A linear layer** `nn.Linear(i, o)` holds a weight of shape `(o, i)` and a bias of shape `(o,)`,
so `o·i + o` parameters, and computes `x @ W.T + b` on the **last** axis: `(B, T, i) → (B, T, o)`
with no reshaping.

**Why activations.** Two linear layers in a row are still one linear function. Between them goes
a simple nonlinear function applied to each number: **ReLU** is `max(0, z)`. Linear → ReLU →
Linear → … → Linear is a **multi-layer perceptron (MLP)**. With enough width it can approximate
almost any function.

**Counting parameters** is a habit. An MLP 4 → 16 → 3 has `(4·16 + 16) + (16·3 + 3) = 131`.

### nn.Module

A module is a class that holds parameters, may contain other modules, and defines `forward`.

```python
class TwoLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                         # without this line nothing registers
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):                          # x: (N, d_in)
        return self.fc2(F.relu(self.fc1(x)))       # (N, d_out)

net = TwoLayer(4, 16, 3)
net(torch.randn(8, 4)).shape                       # call net(x), never net.forward(x)
sum(p.numel() for p in net.parameters())           # 131
[name for name, _ in net.named_parameters()]       # fc1.weight, fc1.bias, fc2.weight, fc2.bias
```

**Registration.** An attribute that is an `nn.Module` or an `nn.Parameter` is registered: it
shows up in `.parameters()` (what the optimiser updates) and in `.state_dict()` (what gets
saved). A plain Python **list** of layers is not registered: the optimiser never sees those
weights and the model silently does not learn. For a variable number of layers use
`nn.ModuleList([...])`, or `nn.Sequential(*layers)` when they simply run one after the other.

A parameter with `requires_grad=False` is **frozen**: still in the model, not trained.

### Train mode and eval mode

`model.train()` and `model.eval()` flip a flag on every sub-module. Only layers that behave
differently in the two modes care. The main one is **dropout**: in training it sets each value
to zero with probability `p` and scales the others up so the average is unchanged; in eval it does
nothing.

`model.eval()` and `torch.no_grad()` are two different things, and evaluation needs **both**:

| | Changes | Without it |
|---|---|---|
| `model.eval()` | layer behaviour (dropout off) | noisy, pessimistic validation numbers |
| `torch.no_grad()` | no graph recorded | same numbers, wasted memory and time |

Neither one freezes weights. Switch back with `model.train()` after evaluating.

### The optimiser

An optimiser holds the parameters and does yesterday's update for you.

```python
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
opt.zero_grad()      # yesterday's p.grad.zero_()
# loss.backward() goes here
opt.step()           # yesterday's "p -= lr * p.grad", for every parameter
```

**SGD** is exactly yesterday's rule. **Adam** keeps, per parameter, a running average of the
gradient and of its square, and steps by their ratio: parameters with small gradients get
relatively larger steps, so one learning rate works across layers. Adam with `lr` around `1e-3`
is the default for small models.

### The loop

```python
batches = [(torch.randn(8, 4), torch.randn(8, 3)) for _ in range(5)]     # stand-in data
net.train()
for xb, yb in batches:
    pred = net(xb)                    # forward
    loss = F.mse_loss(pred, yb)       # one scalar
    opt.zero_grad()                   # clear old gradients
    loss.backward()                   # new gradients
    opt.step()                        # update
    running = loss.item()             # a Python float for logging, not the tensor
```

The order zero → backward → step is fixed. `loss.item()` matters: a list of loss tensors keeps
every graph alive and fills the memory.

**A mean over batches of different sizes.** The last batch is often smaller. The mean loss over
the dataset is `Σ(batch loss × batch size) / Σ(batch size)`, not the plain average of the batch
losses.

### Two checks before any real training

**Loss at step 0.** Know the first loss before you see it. Regression with targets normalised to
mean 0 and std 1, and a fresh network whose outputs are near 0: the MSE is about `mean(y²) ≈ 1`.
Classification over K classes: about `ln K`. A first loss far from that is a bug that exists
before any learning.

**Overfit one batch.** Train on a single batch for a few hundred steps. The loss must go to
almost zero: a network can always memorise 32 examples. If it does not, the bug is in the code
(shapes, loss, optimiser wiring), not in the data or the method. It is the most useful debugging
habit there is, and interviewers ask for it.

### Traps of the day

- No `super().__init__()`.
- Layers in a plain list.
- Evaluation with dropout on, or with the graph recorded.
- `opt.step()` before `loss.backward()`, or no `zero_grad()`.
- An optimiser built before the model's layers exist, or from the wrong model.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `MLP` | configurable depth, optional dropout | layers, nn.Module, registration |
| `count_params` | all parameters, or only the trainable ones | counting, frozen parameters |
| `train_epoch` | one pass, returns the size-weighted mean loss | the loop |
| `evaluate` | deterministic, no graph | train mode and eval mode |
| `overfit_one_batch` | the sanity check, as a function | two checks |

Run: `uv run pytest weeks/w1-pytorch-basics/d4-modules-training-loop -k "not review"`

---

## Day 5: Behavior cloning on a toy SO-101

Folder: `d5-bc-so101-toy`. Exercise: `w1d5_bc.py`. One extra file, `so101_toy.py`: the data
generator, provided. Read its first 15 lines; do not edit it.

### Why this matters

**Behavior cloning (BC)** is supervised learning on what an operator did: given the observation,
predict the operator's action. It is days 1–4 applied to a robot. The machine learning is easy.
The traps are in the data handling and the evaluation, and they are the same traps that bite real
policies.

### The data

60 toy episodes of 60 to 120 steps. Six dimensions like the real arm: five joint angles in
degrees and a gripper from 0 (closed) to 100 (open).

- **Observation** `(12,)`: current joint state (6) + goal pose (6).
- **Action** `(6,)`: the next commanded joint positions (absolute targets, as in LeRobot).

Each episode is a dict with `obs` of shape `(T, 12)` and `action` of shape `(T, 6)`.

### Trap 1: split by episode, never by frame

Frames 41 and 42 of one episode are almost the same. A random split of **frames** puts one in
training and one in validation. The model then "predicts" validation by remembering training:
the validation loss looks great and means nothing. So you choose which **episodes** go to
validation first, with a fixed seed, and only then cut them into frames.

```python
g = torch.Generator().manual_seed(0)
torch.randperm(10, generator=g)      # a seeded shuffle of 0..9: same order every run
```

### Trap 2: normalisation statistics from the training split only, stored in the model

The mean and std of day 1 are computed on the **training episodes only**. Computed on all the
data, they carry information about validation into training.

They must also travel with the weights. A policy loaded with different statistics, or without
any, outputs wrong joint targets and looks like "a bad policy". The tool is a **buffer**: a
tensor that belongs to the module, is saved in the `state_dict`, moves with `.to(device)`, and is
not trained.

```python
class Scaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean)        # saved and moved, never trained
        self.register_buffer("std", std)
    def forward(self, x):
        return (x - self.mean) / self.std

s = Scaler(torch.zeros(3), torch.ones(3))
list(s.state_dict().keys())                       # ['mean', 'std']
len(list(s.parameters()))                         # 0: the optimiser does not see buffers
```

A good policy module takes **raw** observations and returns **raw** actions. It normalises its
input inside `forward`, predicts in normalised units, and converts back with
`action = pred · std + mean`. The loss is computed in **normalised** action space. In raw units,
the joints with the largest range would dominate the loss and the gripper would be ignored.

A joint that barely moves has a tiny std. A floor on the std (`std.clamp(min=...)`) is safer
than a small `eps`: it stops sensor noise from being blown up.

### Trap 3: beat the cheapest baseline first

A **baseline** is the simplest thing that could work. Here: "hold position", the action equals the
current joint state. It costs nothing and it is not bad, because consecutive commands are close.
A number alone means nothing; "validation MSE 4.1" becomes information only next to "hold
position gets 9.8". If the model does not clearly beat the baseline, nothing else matters.

Measure both in **raw** units, on the **validation** episodes, with no graph.

### Trap 4: save, reload, compare

```python
net = nn.Linear(3, 2)
torch.save({"model": net.state_dict(), "hidden": 256}, "/tmp/ckpt.pt")   # weights + what is needed to rebuild
ckpt = torch.load("/tmp/ckpt.pt", map_location="cpu")
net2 = nn.Linear(3, 2)                                                   # 1. rebuild the same architecture
net2.load_state_dict(ckpt["model"], strict=True)                         # 2. fill it; strict raises on any mismatch
```

A `state_dict` is a dict from names to tensors: parameters **and** buffers. A checkpoint also
needs whatever is required to rebuild the module, such as layer sizes.

`strict=True` raises if a key is missing or unexpected. `strict=False` loads what matches and
returns the rest in a result that is easy to ignore: the model then runs with some layers still
random. After loading, feed the same input to the original and the reloaded model and compare the
outputs. If they differ, every later evaluation measures something else.

### Mini-batches

```python
N = 1000
perm = torch.randperm(N)                         # a new shuffle each epoch
for i in range(0, N, 256):
    idx = perm[i:i + 256]                        # indices of one batch; the last one is smaller
    # xb, yb = obs[idx], act[idx]
```

### What "it works" means today

The test trains for real (10–20 s on CPU) and asks for a validation MSE below 75% of the
baseline's. Print both numbers and look at them. The error will not go to zero: the operator in
`so101_toy.py` is noisy, and `questions.md` asks you where the floor comes from.

A low validation MSE still does not mean the arm will succeed. On the robot, small errors move
the arm to states the operator never visited, where the policy is worse, which moves it further:
**compounding error**. Week 6 measures policies in closed loop for that reason.

### Traps of the day

- Splitting frames instead of episodes.
- Statistics computed on all the data, or kept outside the checkpoint.
- Loss in raw units.
- `strict=False` and an unread result.
- Reporting a model number with no baseline next to it.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `split_by_episode` | seeded shuffle of episode ids, then cut | trap 1 |
| `stack_frames` | chosen episodes → `(N, 12)`, `(N, 6)` | day 1, combining tensors |
| `compute_norm_stats` | per-dimension mean and std, with a floor on std | trap 2 |
| `BCPolicy` | raw in, raw out, stats in buffers | trap 2, day 4 |
| `train_bc` | the loop of day 4, loss in normalised action space | mini-batches, day 4 |
| `raw_mse`, `hold_baseline_mse` | the two numbers to compare | trap 3 |
| `save_policy`, `load_policy` | everything needed to rebuild, strict load | trap 4 |

Run: `uv run pytest weeks/w1-pytorch-basics/d5-bc-so101-toy -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w1` under a timer, AI fully off, then
  `uv run pytest drills/w1 -k p01` (or `p02` … `p05`). Protocol: `drills/README.md`.
- **Reading slot (20 min, read only):** this week's file is in `reading/README.md`, row 1.
- **Tick `PROGRESS.md`** (Week 1 table and retro). Write down the hardest lab: next Monday you
  rewrite its main function from a blank file.

You are done with week 1 when you can do these without notes:

- [ ] say the shape after each of `reshape`, `permute`, `stack`, `cat`, `mean(dim=…)`, `unfold`
- [ ] state the broadcasting rule and give one case where it fails silently
- [ ] write a masked mean with the right denominator
- [ ] derive `dW`, `db`, `dx` for a linear layer with MSE
- [ ] write the training loop and the evaluation loop, and say what each line is for
- [ ] explain: episode split, train-only statistics, buffers, baseline, strict load

## Cheat sheet

| Need | Tool |
|---|---|
| new shape, same order of numbers | `reshape` (`view` if contiguous) |
| reorder the axes | `permute`, `transpose` |
| add / remove a size-1 axis | `unsqueeze`, `[:, None]` / `squeeze` |
| new axis from a list / longer along an axis | `stack` / `cat` |
| remove an axis by averaging | `mean(dim=k)`, `keepdim=True` to keep it as size 1 |
| sliding windows | `unfold(dim, size, step)`, window axis goes last |
| every pair of two sets | `a[:, None] − b[None, :]` |
| one entry per row | `gather(1, idx[:, None]).squeeze(1)` |
| mean over valid items | `(x * mask).sum() / mask.sum()` |
| named-axis product | `torch.einsum("bij,bj->bi", W, x)` |
| gradients | `requires_grad=True`, `loss.backward()`, `.grad` |
| cut the graph / stop recording | `.detach()` / `with torch.no_grad():` |
| one training step | `zero_grad()` → `backward()` → `step()` |
| evaluation | `model.eval()` + `torch.no_grad()` |
| saved, moved, not trained | `register_buffer` |
| save / load | `state_dict()`, `load_state_dict(sd, strict=True)` |

## Optional depth (not needed for the labs)

- Longer version of "Deep learning in ten minutes": `reference/journal/book/00p1-deep-learning-from-zero.md`
- The same PyTorch topics with more API and an autograd engine built from scratch (P2.19):
  `reference/journal/book/00p2-pytorch-for-the-coding-round.md`
- Next week: [`../w2-training-real-data/README.md`](../w2-training-real-data/README.md)
