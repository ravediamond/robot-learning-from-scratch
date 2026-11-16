# Week 3: Attention, transformers, GPT-2

**Everything you need for this week is on this page.** You need weeks 1 and 2 (tensors,
broadcasting, einsum, `nn.Module`, the training loop). Nothing from other folders is required.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [The idea of the week](#the-idea-of-the-week).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-scaled-dot-product-attention),
   [Day 2](#day-2-multi-head-attention-and-cross-attention), [Day 3](#day-3-the-block-positions-image-patches),
   [Day 4](#day-4-gpt-2-from-scratch), [Day 5](#day-5-load-real-weights-generate-train).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w3dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w3dN_*.py` | tests | run, never edit |
| `review_w3dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided helpers: `d3-block-positions/mha_ref.py` (a reference multi-head attention, so day 3
does not depend on your day 2), `d5-gpt2-weights-generate/train_char.py`.

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w3-transformers-gpt2/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w3-transformers-gpt2/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w3-transformers-gpt2/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w3-transformers-gpt2/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-attention` | queries, keys, values, masks | attention by hand, causal and padding masks | a causal mask that hides too much |
| Tue | `d2-multihead-cross` | heads, cross-attention | multi-head attention that matches PyTorch's numbers | heads split along the wrong axis |
| Wed | `d3-block-positions` | residuals, LayerNorm, positions, patches | a pre-norm block, sinusoidal positions, RoPE, patchify | a missing residual |
| Thu | `d4-gpt2-model` | the whole GPT-2 architecture | GPT-2 small, exact names and parameter count | tied weights that are not tied |
| Fri | `d5-gpt2-weights-generate` | loading someone else's weights, sampling | load OpenAI's GPT-2 into your model, generate | temperature applied at the wrong place |

## The idea of the week

A **transformer** works on a sequence of vectors called **tokens**. A token can be a piece of
text, a patch of an image, a joint state or an action. Each has the same size `D`. The model
is a stack of identical **blocks**. In each block, every token first gathers information from
the other tokens (**attention**), then is transformed on its own (a small MLP).

SmolVLA's language model, the vision encoders (ViT, DINOv2, SigLIP), the action expert of a VLA
and the predictor of a JEPA world model are all this one architecture. Learn it once here, on
text, where you can check your work against a published model down to the last number.

All code on this page assumes:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Day 1: Scaled dot-product attention

Folder: `d1-attention`. Exercise: `w3d1_attention.py`.

### The mechanism

Each token produces three vectors:

- a **query** `q`: what am I looking for?
- a **key** `k`: what do I contain?
- a **value** `v`: what do I hand over if selected?

For one query: compare it with every key by dot product (large when two vectors point the same
way), turn the scores into weights that sum to 1 with a softmax, and return the weighted average
of the values. It is a soft dictionary lookup.

```
scores = Q Kᵀ / √d          (B, Tq, Tk)    how well each query matches each key
A      = softmax(scores)    over the keys: each row sums to 1
out    = A V                (B, Tq, dv)
```

**A worked example.** Three tokens, `d = 2`, keys equal to queries: `(1, 0)`, `(0, 1)`, `(1, 1)`;
values `(10, 0)`, `(0, 10)`, `(5, 5)`. Token 1's scores against the three keys are
`(1, 0, 1) / √2 = (0.71, 0, 0.71)`. Softmax gives weights `(0.40, 0.20, 0.40)`. Its output is
`0.40·(10, 0) + 0.20·(0, 10) + 0.40·(5, 5) = (6.02, 3.98)`. Token 1 looked mostly at itself and
at token 3, whose key resembles its query. Redo it on paper once.

**Why divide by √d.** A dot product of two random vectors of size `d` has a spread that grows
like `√d`. Large scores make the softmax put almost all its weight on one key, and its gradient
vanishes. Dividing by `√d` keeps the scores at a constant scale.

**Shapes.** `Tq` queries and `Tk` keys need not be the same number. The key and query sizes
must match (they are multiplied); the value size is free. The transpose that makes `Q Kᵀ` work
on batched tensors swaps the **last two** axes: `k.transpose(-2, -1)`.

### Masks

A mask says which (query, key) pairs are allowed. This course uses booleans where
**True = may attend**, with a shape that broadcasts to `(B, Tq, Tk)`.

Forbidden scores are set to `−inf` **before** the softmax, so their weight is exactly 0:

```python
s = torch.tensor([1.0, 2.0, 3.0])
allowed = torch.tensor([True, True, False])
F.softmax(s.masked_fill(~allowed, float("-inf")), dim=0)     # tensor([0.2689, 0.7311, 0.0000])
```

Setting a forbidden score to 0 is wrong: a score of 0 still gets weight `exp(0) = 1`.

**Causal mask** `(T, T)`. When the model predicts the next token, token `i` must not see tokens
after it. Row `i` allows keys `0..i`, itself included: the lower triangle with the diagonal.

```python
torch.tril(torch.ones(4, 4, dtype=torch.bool))     # lower triangle, diagonal included
torch.triu(torch.ones(4, 4, dtype=torch.bool))     # upper triangle
```

**Key padding mask.** Sequences in a batch have different lengths and are padded. No query may
look at a padded key. From the lengths `(B,)` you want `(B, T)` with True where the position is
smaller than the length; the grid trick of week 1, day 2 does it. It then needs a size-1 axis
where the queries go, so it broadcasts over every query: `(B, 1, T)`.

**Combining** two masks: a pair is allowed only if both allow it, the logical `&`.

**A row with no allowed key.** Softmax over all `−inf` is `0/0 = NaN`, and that `NaN` spreads
through the whole model. Real code must decide what such a row means. In this lab it returns
zeros. `torch.nan_to_num`, `torch.where`, or a mask on the result are the tools.

### The PyTorch reference

```python
q = torch.randn(2, 5, 8); k = torch.randn(2, 6, 8); v = torch.randn(2, 6, 3)
F.scaled_dot_product_attention(q, k, v).shape        # (2, 5, 3)
```

It accepts `attn_mask=` as a boolean with True = attend. You may not use it today; the tests
compare your function against it.

### Traps of the day

- Softmax over the query axis instead of the key axis. Rows must sum to 1.
- Masked scores set to 0, or the mask applied after the softmax.
- A causal mask without the diagonal: the first token can see nothing.
- `k.T` on a batched tensor: it reverses **all** axes.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `causal_mask` | `(T, T)` bool, True where `j ≤ i` | masks |
| `key_padding_mask` | lengths `(B,)` → `(B, 1, T)` bool | masks |
| `combine_masks` | AND of the masks that are not `None` | masks |
| `attention_weights` | `A`, with all-zero rows where nothing is allowed | the mechanism, masks |
| `attention` | `A V` | the mechanism |

Run: `uv run pytest weeks/w3-transformers-gpt2/d1-attention -k "not review"`

---

## Day 2: Multi-head attention and cross-attention

Folder: `d2-multihead-cross`. Exercise: `w3d2_mha.py`.

### Where queries, keys and values come from

Three linear layers applied to the tokens: `q = q_proj(x)`, `k = k_proj(x)`, `v = v_proj(x)`.
They are what the model learns. One more linear layer, `out_proj`, mixes the result.

### Heads

One attention gives each token one way of looking at the others. **Multi-head attention** cuts
each vector of size `D` into `h` slices of size `D/h`, runs one attention per slice in parallel,
and joins the results. Different heads can look for different things. The number of parameters
is the same as with one head.

The whole job is reshaping. Follow it slowly:

```
x (B, T, D)  →  q_proj  →  (B, T, D)
             →  split the LAST axis into (h, D/h)        (B, T, h, D/h)
             →  move the head axis next to the batch     (B, h, T, D/h)
attention on the last two axes, for every (batch, head)  (B, h, T, D/h)
             →  move the head axis back                  (B, T, h, D/h)
             →  merge the last two axes                  (B, T, D)  →  out_proj
```

With heads in axis 1, the attention function of yesterday works unchanged: it treats every
leading axis as a batch axis.

**The silent bug.** The slices must be cut from the **feature** axis. A `reshape` straight from
`(B, T, D)` to `(B, h, T, D/h)` has the right shape and mixes time steps with features. Split
the last axis first, then move axes with `transpose`. Remember day 1 of week 1: after a
`transpose`, `view` fails and `reshape` works.

**Masks with heads.** Scores are now `(B, h, Tq, Tk)`. A mask of shape `(Tq, Tk)` broadcasts as
it is. A mask with a batch axis needs a size-1 axis for the heads: `(B, 1, Tq, Tk)`.

### Cross-attention

Self-attention: queries, keys and values all come from the same sequence. **Cross-attention**:
the queries come from one sequence `x`, the keys and values from another, the **context**.

```
q = q_proj(x)            (B, Tq, D)
k = k_proj(context)      (B, Tk, D)
v = v_proj(context)      (B, Tk, D)
```

The output has one vector per **query**: `(B, Tq, D)`. `Tq ≠ Tk` is normal. In a VLA action
expert, `x` is the action tokens and the context is the features of the vision-language model:
that is how actions "read" the image and the instruction.

### Matching PyTorch exactly

```python
D, h = 32, 4
mha = nn.MultiheadAttention(D, h, batch_first=True)       # batch_first! the default is (T, B, D)
x = torch.randn(2, 5, D)
out, attn = mha(x, x, x)                                  # out (2, 5, 32); attn (2, 5, 5), averaged over heads
mha.in_proj_weight.shape, mha.out_proj.weight.shape       # (96, 32), (32, 32)
```

`nn.MultiheadAttention` stores the three projections stacked in one matrix `in_proj_weight` of
shape `(3D, D)`: rows `0:D` are the query projection, `D:2D` the key, `2D:3D` the value. The test
copies them into your module and asks for the same output. If your numbers match PyTorch's, your
shapes are right; no other test is as strict.

**Mask conventions differ.** In `nn.MultiheadAttention`, a boolean mask means True = **ignore**.
In `F.scaled_dot_product_attention` and in this course, True = **attend**. Say which one you use
whenever you write a mask. It is a classic silent bug.

### Traps of the day

- Heads split along time.
- `view` after `transpose`.
- Cross-attention with keys taken from `x` instead of the context.
- A mask in the opposite convention.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `split_heads` | `(B, T, D)` → `(B, h, T, D/h)` | heads |
| `merge_heads` | `(B, h, T, Dh)` → `(B, T, h·Dh)` | heads |
| `MultiHeadAttention` | `q_proj`, `k_proj`, `v_proj`, `out_proj`; self- or cross-attention; stores the last attention weights | all of the above |

Write the attention itself again inline: it is four lines, and writing it twice is the point.

Run: `uv run pytest weeks/w3-transformers-gpt2/d2-multihead-cross -k "not review"`

---

## Day 3: The block, positions, image patches

Folder: `d3-block-positions`. Exercise: `w3d3_block.py`. Provided: `mha_ref.py`.

### The pieces that make a deep stack trainable

**Residual connection.** Instead of `y = f(x)`, a block computes `y = x + f(x)`: it adds a
correction to its input. The gradient flows straight through the `+ x` path, so many blocks can
be stacked and still train.

**LayerNorm.** For each token, subtract the mean of its `D` features, divide by their standard
deviation, then apply a learned scale and shift. It keeps the numbers in a sensible range at
every depth. It does not depend on the batch, so it behaves the same in training and evaluation.

**The MLP.** After attention, each token goes alone through `Linear(D, 4D)` → GELU →
`Linear(4D, D)`. GELU is a smooth version of ReLU.

### The pre-norm block

```
x = x + attn( ln_1(x) )          residual 1: gather from other tokens
x = x + mlp ( ln_2(x) )          residual 2: transform each token
```

"Pre-norm" means the normalisation is applied to the **input** of each sub-layer, inside the
residual branch. The `x` on the left of each `+` is never normalised: a clean path from the
first block to the last. It trains more stably than the original post-norm arrangement, and
modern models use it.

A property worth knowing: if the last layer of `attn` and the last layer of `mlp` output zero,
the block is the identity. Many models initialise those layers near zero so that a deep stack
starts close to the identity. It is also today's test.

### Positions

Attention treats its input as a **set**: shuffle the tokens and each output is the same, just
shuffled. The model needs to be told the order.

**Sinusoidal** (the original transformer). A fixed table `(T, D)`, added to the tokens:

```
PE[t, 2i]   = sin( t / 10000^(2i/D) )
PE[t, 2i+1] = cos( t / 10000^(2i/D) )
```

Each pair of dimensions is a clock that turns at its own speed: fast for small `i`, slow for
large `i`. Together they give every position a unique pattern. To fill even and odd columns,
slices with a step are the tool: `pe[:, 0::2]` and `pe[:, 1::2]`.

**Learned absolute** (GPT-2). An `nn.Embedding(max_T, D)`: one learned vector per position,
added to the tokens. Simple; the model cannot go beyond `max_T`.

**RoPE, rotary position embedding** (LLaMA and most modern VLMs). Nothing is added to the
tokens. Instead, inside attention, the query and the key are **rotated** by an angle that
depends on their position. Take the vector two dimensions at a time; rotate pair `i` of the token
at position `t` by the angle `t · θᵢ`, with `θᵢ = base^(−2i/d)`:

```
x'[2i]   = x[2i] · cos(tθᵢ) − x[2i+1] · sin(tθᵢ)
x'[2i+1] = x[2i] · sin(tθᵢ) + x[2i+1] · cos(tθᵢ)
```

A dot product between two rotated vectors depends only on the **difference** of their angles.
So the score `q_m · k_n` depends on `m − n`, the relative position, and not on where the pair
sits in the sequence. Rotations also keep the length of a vector unchanged; both facts are
easy to test.

### From an image to tokens

A vision transformer (ViT) cuts the image into square patches of side `p`, flattens each patch
into a vector, and applies one linear layer, the **patch embedding**. A 224×224 image with
`p = 16` gives `14 × 14 = 196` tokens.

```
(B, C, H, W)  →  (B, N, C·p·p)        N = (H/p) · (W/p)
```

Two conventions to respect: patches in **row-major** order (left to right, then top to bottom),
and each patch flattened in `(C, p, p)` order. The route is: split `H` into `(H/p, p)` and `W`
into `(W/p, p)` with a reshape, bring the two patch-grid axes together and the three in-patch
axes together with a `permute`, then merge. Write the shape after each step.

```python
img = torch.arange(16.0).reshape(1, 1, 4, 4)
img.reshape(1, 1, 2, 2, 2, 2).shape        # (B, C, H/p, p, W/p, p) for p = 2
```

### Traps of the day

- No residual: the stack trains badly or not at all, with no error.
- Normalising after the residual (post-norm) when pre-norm is asked.
- Sine and cosine in the wrong columns, or the exponent `2i/D` computed with `i` counting columns.
- Patchify by a plain reshape: right shape, pixels from different patches mixed.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `Block` | pre-norm block with `ln_1`, `attn`, `ln_2`, `mlp`; self- or cross-attention | the pre-norm block |
| `sinusoidal_positions` | the `(T, D)` table | positions |
| `apply_rope` | rotate each pair by `position · θᵢ` | positions |
| `patchify` | `(B, C, H, W)` → `(B, N, C·p·p)` | from an image to tokens |

Run: `uv run pytest weeks/w3-transformers-gpt2/d3-block-positions -k "not review"`

---

## Day 4: GPT-2 from scratch

Folder: `d4-gpt2-model`. Exercise: `w3d4_gpt2.py`.

### Why this matters

GPT-2 is the smallest model that is a real, published transformer with public weights. Tomorrow
you load those weights into **your** code, so today every name and every shape must match the
original. Building to an exact external reference is the daily job with VLAs.

### Language modelling

Text is cut into **tokens** and each token is an integer id from a vocabulary of `V` entries.
The task: at every position, predict the next token. An input of `T` ids gives `T` predictions
at once, each a vector of `V` scores (logits). Position `t` must not see position `t + 1`,
which is its own answer: the attention is **causal**.

### The architecture

GPT-2 small: `L = 12` blocks, `D = 768`, `h = 12` heads, `V = 50257`, at most `1024` positions.

```
idx (B, T)  →  wte(idx) + wpe(arange(T))                            (B, T, D)
for each block in h:
    x = x + attn.c_proj( causal_attention( attn.c_attn( ln_1(x) ) ) )
    x = x + mlp.c_proj( gelu( mlp.c_fc( ln_2(x) ) ) )
x = ln_f(x)
logits = lm_head(x)                                                  (B, T, V)
```

- `wte`: token embedding, `nn.Embedding(V, D)`. An embedding is a lookup table: ids of any shape
  in, vectors of that shape plus `D` out.
- `wpe`: learned position embedding, `nn.Embedding(1024, D)`. The positions `(T, D)` broadcast
  over the batch.
- `c_attn`: **one** `Linear(D, 3D)` that produces query, key and value side by side. Split its
  output into three chunks of size `D` along the last axis (`.split(D, dim=-1)` or `.chunk(3, dim=-1)`).
- `c_proj` in the attention: the output projection, `Linear(D, D)`.
- `c_fc`, `c_proj` in the MLP: `Linear(D, 4D)` and `Linear(4D, D)`.
- GELU in its tanh approximation: `F.gelu(x, approximate="tanh")`. LayerNorm `eps = 1e-5`.
- `ln_f`: one last LayerNorm after the stack, because pre-norm blocks leave their output un-normalised.
- `lm_head`: `Linear(D, V, bias=False)`, from vectors to logits.

You wrote attention by hand on Monday, so today you may use
`F.scaled_dot_product_attention(q, k, v, is_causal=True)` on tensors of shape `(B, h, T, D/h)`.

### Weight tying

The input embedding maps an id to a vector; the output layer maps a vector to a score per id.
Both are `(V, D)` matrices about the same vocabulary. GPT-2 uses **one** matrix for both. That
removes 38.6 million parameters, 31% of the model.

It must be the **same tensor object**, not a copy with equal values:

```python
emb = nn.Embedding(10, 4)
head = nn.Linear(4, 10, bias=False)
head.weight = emb.weight                  # one Parameter, two names
head.weight is emb.weight                 # True
sum(p.numel() for p in nn.ModuleList([emb, head]).parameters())     # 40, counted once
```

A copy (`.clone()`, or assigning `.data`) gives equal values at step 0 and two matrices that
drift apart during training. `parameters()` counts a shared tensor once, which is why the
parameter count is a test of tying.

### Initialisation

All Linear and Embedding weights are drawn from `N(0, 0.02)`, biases are zero. The two `c_proj`
of each block use a smaller standard deviation, `0.02 / √(2L)`: each block adds two residual
contributions, `2L` in total, and the scaling keeps the sum at a constant size whatever the depth.

```python
lin = nn.Linear(4, 4)
nn.init.normal_(lin.weight, mean=0.0, std=0.02)       # in place, handles no_grad itself
nn.init.zeros_(lin.bias)
```

`model.apply(fn)` calls `fn` on every sub-module; `named_parameters()` lets you pick parameters
by the end of their name.

### The loss

`forward(idx, targets)` returns `(logits, loss)`. Cross-entropy wants `(N, C)` logits and `(N,)`
targets, so both are flattened over batch and time: `(B·T, V)` and `(B·T,)`. `ignore_index=-1`
skips positions whose target is −1. The targets are already shifted by the caller.

The step-0 check of week 2: a fresh model should give a loss near `ln V = ln 50257 ≈ 10.8`.

### The pass mark

GPT-2 small has exactly **124,439,808** parameters. Count each piece on paper before running
(embeddings, one block, the final norm), and find your mistake by comparing piece by piece.
A dataclass is the usual way to hold the configuration:

```python
from dataclasses import dataclass

@dataclass
class Cfg:
    n_layer: int = 2
    n_embd: int = 64
```

### Traps of the day

- A module name that differs from the original by one character: tomorrow's load fails.
- `lm_head` with a bias, or tied by copy.
- The position embedding indexed with token ids.
- Logits not flattened for the loss.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `GPT2Config` | the numbers above | the architecture |
| `CausalSelfAttention` | `c_attn`, split, heads, causal attention, `c_proj` | the architecture, days 1–2 |
| `MLP` | `c_fc`, GELU (tanh), `c_proj` | the architecture |
| `GPT2Block` | `ln_1`, `attn`, `ln_2`, `mlp`, pre-norm | day 3 |
| `GPT2` | `wte`, `wpe`, `h`, `ln_f`, tied `lm_head`, init, `(logits, loss)` | weight tying, initialisation, the loss |

Run: `uv run pytest weeks/w3-transformers-gpt2/d4-gpt2-model -k "not review"`

---

## Day 5: Load real weights, generate, train

Folder: `d5-gpt2-weights-generate`. Exercise: `w3d5_load_generate.py`. It imports your day-4
model automatically. Provided: `train_char.py`.

### Loading someone else's weights

This is the job you will do for every VLA checkpoint. A `state_dict` is a dict from names to
tensors. Yours and theirs describe the same model with different names and layouts. Five steps:

1. **Map the names.** Hugging Face's GPT-2 prefixes everything with `transformer.`; yours has no prefix.
2. **Fix the layouts.** OpenAI's code used a layer called `Conv1D` that stores its weight as
   `(in, out)`. `nn.Linear` stores `(out, in)`. Four weights per block must be transposed with
   `.t()`: `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`. Their biases are unchanged.
3. **Skip what is not a parameter.** Old checkpoints store the causal mask as buffers
   (`...attn.bias`, `...attn.masked_bias`). They are not weights.
4. **Count.** Every one of your tensors must be filled exactly once, and shapes must be equal. A
   silent partial load gives a model that runs and produces garbage. Your tied `lm_head.weight`
   is the same tensor as `wte.weight`: fill it once.
5. **Check the outputs.** Same input, same logits as the reference, to about `1e-4`. Only then
   trust it.

Copying into a parameter is an in-place change of a leaf, so it goes inside `torch.no_grad()`:

```python
dst = nn.Linear(3, 2)
src = torch.randn(3, 2)                    # stored as (in, out)
with torch.no_grad():
    dst.weight.copy_(src.t())              # copy_ checks nothing about meaning; compare shapes yourself first
```

A square weight, such as `attn.c_proj` at `(768, 768)`, has the same shape transposed or not.
A shape check cannot catch a missing transpose there; only the output comparison of step 5 can.

### Generation

The model gives logits for every position; only the **last** one predicts the token that comes next.

```
repeat:
    logits = model(idx[:, -n_positions:])[0][:, -1]      (B, V): last position only
    logits = logits / temperature
    keep the k largest logits, set the others to −inf
    probs  = softmax(logits)
    next   = one sample from probs                        (B, 1)
    idx    = cat(idx, next)
```

- **Crop the context** to the model's maximum: there is no position embedding beyond it.
- **Temperature** divides the **logits**, before the softmax. Below 1 the distribution gets
  sharper, above 1 flatter. Temperature 0 means: take the argmax (greedy), no division.
- **Top-k** removes the unlikely tail before sampling. `logits.topk(k)` gives the `k` largest
  values per row; everything below the smallest of them is masked.
- `torch.multinomial(probs, num_samples=1, generator=g)` draws one index per row; with a seeded
  generator the text is reproducible.

Each new token recomputes the whole prefix. A **KV cache** stores the keys and values of past
tokens so each step only computes the new one; `questions.md` asks you how.

### Data for a character-level model

A tokenizer maps text to ids and back. The simplest: one id per distinct character.

```python
text = "hello"
chars = sorted(set(text))                      # ['e', 'h', 'l', 'o']
stoi = {c: i for i, c in enumerate(chars)}     # char → id
```

A training batch is `batch` random windows of `block` ids, and the same windows shifted by one
as targets: `y[b, t]` is the token that follows `x[b, t]`. One window gives `block` training
signals. `torch.randint(low, high, (n,), generator=g)` draws the start positions; `high` is exclusive.

### After the tests pass, on your Mac

```bash
GPT2_REAL=1 uv run pytest weeks/w3-transformers-gpt2/d5-gpt2-weights-generate -k real -s   # downloads 500 MB once
uv run python weeks/w3-transformers-gpt2/d5-gpt2-weights-generate/train_char.py            # ~10 min on MPS
```

The first loads OpenAI's real GPT-2 into your model and generates text. The second trains a
small GPT on your own book and prints samples.

### Traps of the day

- A missing transpose on a square matrix.
- Tied weight filled twice, or counted twice.
- Temperature applied to probabilities.
- Sampling from every position instead of the last.
- A window start that lets `y` run past the end of the data.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `load_hf_state_dict` | names, transposes, skips, count, errors | loading someone else's weights |
| `generate` | crop, last position, temperature, top-k, sample or argmax | generation |
| `CharTokenizer` | characters ↔ ids | data for a character-level model |
| `get_batch` | random windows and their shifted targets | data for a character-level model |

Run: `uv run pytest weeks/w3-transformers-gpt2/d5-gpt2-weights-generate -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w3` (heaps, top-k), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 3 (SmolVLA's VLM with its expert).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 3 when you can do these without notes:

- [ ] write scaled dot-product attention with a causal and a padding mask, and say every shape
- [ ] write split-heads and merge-heads, and explain the bug of splitting along time
- [ ] draw a pre-norm block and say why residuals and why pre-norm
- [ ] explain sinusoidal, learned and rotary positions in one sentence each
- [ ] say what weight tying is and how to test it
- [ ] list the five steps of loading foreign weights
- [ ] write the generation loop with temperature and top-k

## Optional depth (not needed for the labs)

- Interview questions on this week, with short answers: `knowledge/01-transformers-attention.md`
- A complete tiny GPT with training, explained line by line: `reference/journal/book/00p2-pytorch-for-the-coding-round.md` (P2.20)
- Timed versions for later: `drills/live/L01_causal_mha.py`, `L05_rope.py`, `L08_block.py`
- Previous week: [`../w2-training-real-data/README.md`](../w2-training-real-data/README.md) · Next: [`../w4-generative-flow/README.md`](../w4-generative-flow/README.md)
