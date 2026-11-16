# Week 5: Vision-language models and vision-language-action models

**Everything you need for this week is on this page.** You need week 3 (attention, the block,
patchify) and week 4 (flow matching). Nothing from other folders is required.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [The idea of the week](#the-idea-of-the-week).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-clip-and-siglip),
   [Day 2](#day-2-vlm-plumbing), [Day 3](#day-3-action-tokenizers),
   [Day 4](#day-4-lora-and-what-to-freeze), [Day 5](#day-5-a-mini-vla).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w5dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w5dN_*.py` | tests | run, never edit |
| `review_w5dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided helpers: `d1-clip-siglip/shapes_data.py` (images of shapes with captions),
`d5-mini-vla/vla_world.py` (the task) and `vla_blocks.py` (week 3's attention and block).

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w5-vlm-vla/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w5-vlm-vla/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w5-vlm-vla/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w5-vlm-vla/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-clip-siglip` | contrastive learning of images and text | both losses, a tiny CLIP, zero-shot classification | embeddings that are not normalised |
| Tue | `d2-vlm-plumbing` | how image tokens reach a language model; attention blocks of a VLA | pixel shuffle, projector, block mask, prefix assembly | merging tokens without reordering axes |
| Wed | `d3-action-tokenizers` | actions as tokens: bins, DCT | both tokenizers | decoding to the wrong point of a bin |
| Thu | `d4-lora` | low-rank adaptation, freezing | LoRA from scratch | an initialisation that cannot learn |
| Fri | `d5-mini-vla` | a full VLA, end to end | a small VLA that follows an instruction | the wrong kind of time given to the network |

## The idea of the week

A **VLA** (vision-language-action model) is three things put together:

```
images ──► vision encoder ──► image tokens ─┐
instruction ──► text embedding ─► text tokens ─┼─► language model ──► features ──► action expert ──► action chunk
robot state ─────────────────► state token ──┘      (a transformer)                (flow matching)
```

1. A **vision encoder**, pretrained so that its features already mean something in words (Monday).
2. A **language model** that reads image tokens and text tokens as one sequence (Tuesday). The
   vision encoder plus the language model is a **VLM**.
3. An **action head**. Either the language model writes actions as tokens (Wednesday), or a
   separate **action expert** generates them with flow matching while reading the VLM's features (Friday).

The VLM is pretrained on internet-scale data and is expensive to change. How much of it you train
on robot data is a design decision (Thursday).

All code on this page assumes:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Day 1: CLIP and SigLIP

Folder: `d1-clip-siglip`. Exercise: `w5d1_clip.py`. Provided: `shapes_data.py`.

### The idea

Take a huge set of (image, caption) pairs. Train an image encoder and a text encoder so that an
image and **its own** caption get similar vectors, and an image and **another** caption get
different vectors. No labels are needed. This is **contrastive learning**.

The result is an image encoder whose features line up with language. That is why it is the
vision tower of nearly every VLM and VLA (SigLIP in SmolVLA and π0).

### Similarity

Both encoders output a vector of size `d`. Similarity is the **cosine**: the dot product of the
two vectors after each is scaled to length 1.

```python
a = torch.randn(4, 16)
z = F.normalize(a, dim=-1)                   # each row now has length 1
z.norm(dim=-1)                               # tensor([1., 1., 1., 1.])
```

For a batch of `N` images and their `N` captions, one matrix product gives every pair:
`S = z @ u.T`, shape `(N, N)`. Row `i` is image `i` against all captions. The **diagonal** holds
the true pairs.

Without the normalisation, the model can lower the loss by making vectors longer, which says
nothing about direction. That is today's review bug.

Cosines lie in [−1, 1], too narrow a range for a softmax to become confident. So the matrix is
multiplied by a learned **scale** (a temperature). It is stored as its logarithm and passed
through `exp`, which keeps it positive.

### The CLIP loss

Read row `i` of `S` as class scores: "which of the `N` captions belongs to image `i`?". The
right answer is class `i`. That is a cross-entropy with targets `0, 1, …, N−1`. Then the same
question for each caption against all images: the same cross-entropy on `S.T`. The loss is the
average of the two.

```python
S = torch.randn(5, 5)
F.cross_entropy(S, torch.arange(5))          # image → caption; do it on S.T as well
```

The softmax runs over the batch, so every other item of the batch is a negative. Large batches
give more and harder negatives; CLIP was trained with batches of 32 768.

### The SigLIP loss

Each of the `N²` pairs becomes its own yes/no question: "do this image and this caption
match?". Labels are `+1` on the diagonal and `−1` elsewhere.

```
logits = z @ u.T · t + b                       t: learned scale, b: learned bias
loss   = − mean over all N² pairs of  logsigmoid( labels · logits )
```

`logsigmoid(label · logit)` is the log-probability of the right answer for one pair. There is no
softmax over the batch, so the loss does not need huge batches. The bias `b` starts negative
because almost all pairs are negatives.

```python
labels = 2 * torch.eye(5) - 1                # +1 on the diagonal, −1 elsewhere
F.logsigmoid(torch.tensor([3.0, -3.0]))      # near 0 when confident and right, very negative when wrong
```

### False negatives

If two images of a batch have the same caption, CLIP treats each one as a negative for the
other: it is told that a correct match is wrong. Today's data has only 24 distinct captions, so
it happens in every batch. Watch what it does to the two losses.

### Zero-shot classification

"Zero-shot" means classifying without training a classifier. Encode the image. Encode one
caption per class. The predicted class is the caption with the highest cosine similarity.
Accuracy is the fraction of images whose best caption is their true one. This is how a
contrastive model is evaluated, and how it is used to label data.

### The tiny model

- **Image tower**: a small CNN (week 2). `nn.MaxPool2d(2)` halves the resolution by keeping the
  largest value of each 2×2 block; `nn.AdaptiveAvgPool2d(2)` averages down to a 2×2 map whatever
  the input size; `nn.Flatten()` turns `(B, C, 2, 2)` into `(B, 4C)`.
- **Text tower**: a **bag of words**. Each caption is 3 token ids; their embeddings are summed,
  so word order is ignored. Enough for this data.

### Traps of the day

- No normalisation before the similarity.
- The scale used as stored, without `exp`.
- Cross-entropy in one direction only.
- Evaluating zero-shot against the captions of the batch instead of all the class captions.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `clip_loss` | normalise, scale, symmetric cross-entropy | the CLIP loss |
| `siglip_loss` | normalise, scale and bias, mean log-sigmoid over all pairs | the SigLIP loss |
| `TinyCLIP` | CNN image tower, bag-of-words text tower, learned scale | the tiny model |
| `train_clip` | the loop | weeks 1–2 |
| `zero_shot_accuracy` | best caption among all class captions | zero-shot classification |

Run: `uv run pytest weeks/w5-vlm-vla/d1-clip-siglip -k "not review"`

---

## Day 2: VLM plumbing

Folder: `d2-vlm-plumbing`. Exercise: `w5d2_plumbing.py`.

### From an image to tokens a language model can read

A ViT (week 3, day 3) turns an image into `N = h × w` patch features of size `C`. Two problems
before a language model can use them: there are too many, and they have the wrong size.

```
ViT patch features   (B, N, C)
  → pixel shuffle r  (B, N / r², C · r²)        fewer tokens, each one wider
  → projector MLP    (B, N / r², D_lm)          the language model's token size
```

**Pixel shuffle** (also called token merging here) groups each `r × r` block of neighbouring
tokens into **one** token by concatenating their features. No number is lost: they are moved
from the grid into the channels. SmolVLM goes from 1024 to 64 tokens per image with `r = 4`.
Fewer tokens means a much cheaper language model, since attention cost grows with the square
of the sequence length.

It is week 3's `patchify`, applied to a grid of tokens instead of pixels: restore the 2-D grid
`(B, h, w, C)`, split `h` into `(h/r, r)` and `w` into `(w/r, r)`, bring the two block axes
next to each other with a `permute`, merge. The docstring fixes the order inside a merged token:
row in block, column in block, channel. A reshape without the permute has the right shape and
mixes tokens from different blocks: today's review bug.

**The projector** is a small MLP that maps the merged features to the size of the language
model's tokens. It is the only new part when a vision encoder and a language model are joined,
and often the first thing trained.

### The sequence of a VLA and who may look at whom

The transformer receives one sequence made of blocks:

```
[ image tokens + text tokens | state token | action tokens ]
          block 0                  block 1        block 2
```

The attention mask (True = may attend, week 3) is built from one rule: **a token attends every
token of its own block and of the blocks before it, and nothing after.**

- Block 0 (images and text): each token sees all of block 0, in both directions. It never sees
  the state or the actions.
- Block 1 (state): sees block 0 and itself.
- Block 2 (actions): sees everything before. Inside the block, either every action token sees
  every other (π0), or the block is causal inside itself (SmolVLA's self-attention layers).

**Why this shape.** Block 0, the **prefix**, does not depend on the actions. Sampling an action
chunk with flow matching calls the action expert about 10 times. With this mask the expensive
prefix is computed **once** per observation and reused by all 10 calls. Without it, the whole
VLM would run 10 times.

To build the mask, give every position its block number and compare, with the grid trick of
week 1: one tensor of block ids as a column, the same as a row.

```python
block_of = torch.tensor([0, 0, 0, 1, 2, 2])             # block id of each position
block_of[:, None], block_of[None, :]                    # (T, 1) queries, (1, T) keys
```

### Assembling the prefix

Several cameras, one instruction, one state. Each is already a tensor of tokens of size `D`.
The prefix is their concatenation along the token axis, in a fixed order.

Instructions have different lengths and are padded, so the text comes with a mask (False on
padding). The assembled sequence needs a **key mask** of its own length: True for every image
and state token, the text mask for the text positions. It is built with the same `cat`, on
masks instead of tokens.

### Traps of the day

- Pixel shuffle by reshape only.
- A block mask in the opposite convention, or one that lets the prefix see the actions.
- The key mask and the sequence concatenated in different orders.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `pixel_shuffle_tokens` | `(B, N, C)` → `(B, N/r², C·r²)` | from an image to tokens |
| `Projector` | Linear → GELU → Linear | from an image to tokens |
| `block_attention_mask` | `(T, T)` bool from block sizes, optional causal last block | the sequence of a VLA |
| `assemble_prefix` | cameras, text, state → one sequence and its key mask | assembling the prefix |

Run: `uv run pytest weeks/w5-vlm-vla/d2-vlm-plumbing -k "not review"`

---

## Day 3: Action tokenizers

Folder: `d3-action-tokenizers`. Exercise: `w5d3_tokenizers.py`.

### Why tokenise actions

A language model outputs tokens: integers from a vocabulary. If actions are turned into
integers, the same model can output actions with no new head, trained with the same
cross-entropy. RT-2 and OpenVLA do this. The question is how to turn continuous joint values
into integers, and how many integers a chunk costs.

### Uniform bins

Cut the range `[low, high]` of each action dimension into `n` equal bins. A value becomes the
index of its bin. Decoding returns the **centre** of the bin.

```
width = (high − low) / n
bin b covers [low + b · width, low + (b + 1) · width)
```

The largest error of a round trip is half a bin width: with 256 bins over a range of 2, that is
0.004. Values outside the range are clamped first, and the value `high` itself must land in the
last bin, not in a bin that does not exist.

```python
torch.floor(torch.tensor([0.0, 0.49, 0.5, 0.99]) / 0.5).long()     # which half each value is in
torch.tensor([-3, 0, 7]).clamp(0, 3)                               # keep indices in range
```

The cost: a chunk of `H = 50` steps and `D = 6` joints is 300 tokens, generated one after the
other. That is slow, and wasteful: a smooth motion makes consecutive tokens nearly identical.

### DCT compression (the core of FAST)

A smooth signal can be described with far fewer numbers than it has samples. The **discrete
cosine transform (DCT)** rewrites a signal of `H` samples as `H` coefficients, each the weight
of a cosine wave of a given frequency. Coefficient 0 is the average level; coefficient 1 a
half-wave; higher ones faster wiggles. For a smooth trajectory almost all the energy is in the
first few.

The transform is a matrix product with a fixed `H × H` matrix:

```
C[k, n] = s_k · cos( π · (n + 0.5) · k / H )        k: frequency (row), n: time (column)
s_0 = sqrt(1 / H),   s_k = sqrt(2 / H) for k > 0
```

With those scales `C` is **orthonormal**: its transpose is its inverse. So

```
coeff = C @ chunk          (H, D): one column of coefficients per joint
chunk = C.T @ coeff        exact reconstruction
```

To compress: keep only the first `K` rows of coefficients, multiply by a scale and round to
integers. Those `K × D` integers are the tokens. To decode: divide by the scale, and multiply
by the transpose of the first `K` rows.

```python
k = torch.arange(4.0)[:, None]         # (H, 1): frequencies as a column
n = torch.arange(4.0)[None, :]         # (1, H): times as a row
(n + 0.5) * k                          # (H, H): the grid inside the cosine
```

Two sources of error: the dropped high frequencies (small for smooth motion) and the rounding
(set by the scale). FAST then compresses the integers further with byte-pair encoding, the
method text tokenizers use; this lab stops before that.

### Tokens or a flow head?

| | Tokens (RT-2, OpenVLA, π0-FAST) | Flow head (π0, SmolVLA) |
|---|---|---|
| Training | cross-entropy, same as language; trains fast | flow-matching loss, a separate expert |
| Inference | one token at a time: slow for long chunks | about 10 parallel steps for the whole chunk |
| Precision | limited by bins or rounding | continuous |
| Multi-modal actions | yes, by sampling tokens | yes, by sampling noise |

### Traps of the day

- Decoding to the lower edge of the bin: every action is biased by half a bin.
- `high` mapped to bin `n`.
- `round` on a tensor that is then left as float: tokens must be `long`.
- Rows and columns of the DCT matrix swapped.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `bin_encode`, `bin_decode` | values ↔ bin indices, decode to centres | uniform bins |
| `dct_matrix` | the orthonormal `(H, H)` matrix | DCT compression |
| `dct_encode`, `dct_decode` | chunk ↔ `(keep, D)` integer tokens | DCT compression |

Run: `uv run pytest weeks/w5-vlm-vla/d3-action-tokenizers -k "not review"`

---

## Day 4: LoRA, and what to freeze

Folder: `d4-lora`. Exercise: `w5d4_lora.py`.

### The problem

A pretrained VLM has hundreds of millions to billions of weights and knows a lot. Fine-tuning
all of it on a few hundred robot episodes is expensive and makes it **forget** what it knew.
Training none of it may leave it unable to adapt. **LoRA** (low-rank adaptation) is the middle
road: the pretrained weights stay frozen, and a small number of new weights learn a correction.

### Low-rank update

A linear layer has a weight `W` of shape `(out, in)`. Instead of changing `W`, learn an update
`ΔW = B A` made of two thin matrices:

```
A : (r, in)       B : (out, r)       r much smaller than in and out
y = x Wᵀ + b  +  (alpha / r) · (x Aᵀ) Bᵀ
```

`B A` has shape `(out, in)` but only `r · (in + out)` numbers. For a `768 × 768` layer and
`r = 8`: 12 288 trainable numbers against 589 824. "Low rank" means the update can only move the
layer's output along `r` directions; in practice that is enough to adapt.

`alpha / r` is a fixed scale, so that changing `r` does not change the size of the update much.

### The initialisation

```
A ~ N(0, 1/r)    small random
B = 0
```

With `B = 0` the update is zero: at the start the model is **exactly** the pretrained one. That
is the point of LoRA, and it is testable: same output before and after wrapping.

Why not both zero? Follow the gradients. The gradient of `B` is proportional to `x Aᵀ`: zero if
`A` is zero. The gradient of `A` passes through `B`: zero if `B` is zero. With both at zero, both
gradients are zero and stay zero: nothing ever learns. One of the two must be non-zero, and
choosing `B = 0` keeps the update at zero. Today's review bug is this.

### Freezing and wrapping

```python
lin = nn.Linear(4, 3)
for p in lin.parameters():
    p.requires_grad_(False)                          # frozen: no gradient, the optimiser must skip it
[p.requires_grad for p in lin.parameters()]          # [False, False]
```

A new tensor becomes a trainable part of a module through `nn.Parameter(tensor)`.

To add LoRA to an existing model, you replace chosen `nn.Linear` sub-modules by a wrapper that
contains the original. A sub-module is addressed by its dotted name, as `named_modules()` lists
it (`blocks.1.attn.q_proj`). Replacing it means setting the attribute on its **parent**:

```python
model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
[name for name, _ in model.named_modules()]          # '', '0', '1', '2'
parent = model.get_submodule("")                     # '' is the model itself; 'blocks.1.attn' for a deeper one
setattr(parent, "2", nn.Linear(4, 2))                # replace the child named '2'
```

A dotted name splits into the parent's name and the child's name at the last dot
(`name.rpartition(".")`).

### Merging

After training, the update can be folded into the weight: `W' = W + (alpha / r) · B A`. The
merged layer is a plain `nn.Linear` with the same output. Inference then costs nothing extra.

### What to freeze in a VLA

| Strategy | Cost | Risk |
|---|---|---|
| frozen VLM, trained action expert | lowest | the VLM's features may not carry what control needs |
| LoRA on the language model, trained expert | low | a middle road; the usual first try |
| full fine-tune | highest | forgets language, overfits small robot data |
| vision tower frozen | - | with little data, almost always the safe choice |

It is a decision to take, then to measure. Your own REKI work is the frozen-VLM case.

### Traps of the day

- `A` and `B` both zero.
- The base layer left trainable, or its bias forgotten when freezing.
- `A` and `B` created as plain tensors: not registered, not trained.
- Replacing a module in a loop over `named_modules()` while iterating it: collect the names first.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `LoRALinear` | frozen `base`, `A`, `B`, `scale`, `merged()` | low-rank update, initialisation, merging |
| `apply_lora` | freeze everything, wrap the named Linear layers | freezing and wrapping |
| `trainable_params` | count what still trains | freezing |

Run: `uv run pytest weeks/w5-vlm-vla/d4-lora -k "not review"`

---

## Day 5: A mini VLA

Folder: `d5-mini-vla`. Exercise: `w5d5_mini_vla.py`. Provided: `vla_world.py` (read it first),
`vla_blocks.py` (week 3's `MultiHeadAttention` and `Block`).

### The task

Two targets, one red and one blue, at random places. An instruction: "go to the red" or "go to
the blue". The **image** says where the targets are. The **instruction** says which one. A
policy that ignores either fails half the time.

That makes the question "does it use language?" measurable: give the same image with the other
instruction and see where the robot goes. Both must work.

### The architecture, SmolVLA's layout at toy size

**The prefix** (computed once per observation):

```
image (B, 3, 16, 16) → patchify, p = 4 → 16 tokens of 48 numbers → Linear(48, D)  +  Linear(2, D)(patch centre xy)
instruction (B, 4) ids → Embedding(6, D)  +  learned positions (4, D)
state (B, 2) → Linear(2, D)                                                        one token
all 21 tokens → 2 Blocks, every token sees every token → LayerNorm → prefix (B, 21, D)
```

Each kind of input gets its own small layer that maps it to tokens of size `D`. Then they are
one sequence, and attention mixes them: this is where the instruction "red" meets the red patch.

Patch positions come from a linear layer on the **coordinates** of each patch centre, not from
a learned table. With so little training, that makes "where is it" far easier to learn.

**The action expert** (called at every flow step):

```
noisy chunk x_t (B, H, 2) → Linear(2, D)  +  learned action positions (H, D)  +  Linear(32, D)(time_embedding(t))
→ Block with cross-attention: action tokens are the queries, the prefix is the context
→ Block with causal self-attention over the H action tokens
→ Linear(D, 2) = velocity (B, H, 2)
```

The expert is week 4's velocity network, with a transformer in place of the MLP. It learns
about the scene only through the **cross-attention** to the prefix (week 3, day 2). The time
embedding has shape `(B, D)` after its linear layer and is added to every action token: give it
a token axis, `[:, None, :]`.

**Training** is flow matching on chunks, with `t = 0` noise and `t = 1` data (week 4, day 3).
`t` is a real number in [0, 1) drawn per sample. The network must receive that `t`, through
`time_embedding`. Feeding it the integer index of a sampling step instead is today's review bug.

**Acting**: encode the prefix once, start from noise `(B, H, 2)`, take 10 Euler steps, each
calling only the expert with the cached prefix. That reuse is the reason for Tuesday's mask.

### The training recipe

`train_vla` uses everything from week 2, day 2: AdamW with weight decay 0.01, a linear warm-up
over 200 steps, gradient clipping at 1.0. Transformers are less forgiving than MLPs; without
warm-up and clipping this one often diverges.

### Reading the result

Two tests, about two minutes in total: success above 85% with the given instruction, **and**
above 85% with the instruction swapped. The second is the one that matters. A model at 50% on it
has learned to go to one colour, or to the nearest target, and ignores the text.

### Traps of the day

- The prefix recomputed at every Euler step: correct, ten times slower.
- A mask that makes the cross-attention causal. It is the action self-attention that is causal.
- Time embedding not broadcast over the action tokens.
- Chunks left in raw units; the docstring gives the training units.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `MiniVLA.encode_prefix` | image, instruction, state → `(B, 21, D)` | the prefix |
| `MiniVLA.expert` | `x_t`, `t`, prefix → velocity | the action expert |
| `MiniVLA.forward` | the flow-matching loss on a batch | training |
| `MiniVLA.act` | cached prefix, Euler steps → a chunk | acting |
| `train_vla` | AdamW, warm-up, clipping | the training recipe |

Run: `uv run pytest weeks/w5-vlm-vla/d5-mini-vla -k "not review" -s` (about 2 minutes)

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w5` (union-find, topological sort, Dijkstra, cycles), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 5 (LeRobot's training script).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 5 when you can do these without notes:

- [ ] write the CLIP and the SigLIP loss and say when each is better
- [ ] explain pixel shuffle and the projector, with shapes
- [ ] draw the block attention mask of a VLA and say why the prefix can be cached
- [ ] compare action tokens with a flow head on speed, precision and training
- [ ] write the LoRA forward, its initialisation and why, and the merge
- [ ] draw a VLA from image, text and state to an action chunk, and name the test that proves it uses language

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/02-vlm.md`, `knowledge/03-vla.md`
- SmolVLA and other policies in depth: `reference/journal/book/11a-vla-policy-papers-in-depth.md`
- Previous week: [`../w4-generative-flow/README.md`](../w4-generative-flow/README.md) · Next: [`../w6-robot-learning-eval/README.md`](../w6-robot-learning-eval/README.md)
