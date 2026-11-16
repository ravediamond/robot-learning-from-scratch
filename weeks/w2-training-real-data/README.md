# Week 2: Training properly, CNN encoders, your real SO-101 data

**Everything you need for this week is on this page.** You need week 1 (tensors, autograd, the
training loop, behavior cloning). Nothing from other folders is required.

## Start here

1. Read [The week in one table](#the-week-in-one-table).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-datasets-loaders-episode-windows),
   [Day 2](#day-2-training-that-does-not-blow-up), [Day 3](#day-3-cnn-encoders-for-policies),
   [Day 4](#day-4-your-real-so-101-data-by-hand), [Day 5](#day-5-a-pixel--state-policy).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w2dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w2dN_*.py` | tests | run, never edit |
| `review_w2dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Other `.py` files in a folder are provided helpers (listed in each day below). Read them, do not edit them.

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall of last week's hardest function instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w2-training-real-data/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w2-training-real-data/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w2-training-real-data/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w2-training-real-data/<day>` |

From this week on, some `predict_run.py` scripts import **your** code: run them after your tests pass.

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-datasets-loaders` | `Dataset`, `DataLoader`, collate, windows over episodes | a window dataset with padding and a mask, a seeded loader, balanced sampling | windows that cross episode boundaries |
| Tue | `d2-training-stability` | step-0 loss, warm-up and cosine, gradient clipping, hooks | each of those by hand | clipping at the wrong moment |
| Wed | `d3-cnn-encoders` | convolution, receptive field, spatial softmax, GroupNorm, random shift | a conv from `unfold`, a policy image encoder | image axes in the wrong order |
| Thu | `d4-real-so101-data` | what a LeRobot dataset is on disk | read your own dataset with pandas, check it | rows not sorted in time |
| Fri | `d5-pixel-policy-real` | fusing image and state, chunk outputs, ablation | a pixel + state chunk policy, then trained on your data | augmentation at evaluation |

All code on this page assumes:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
```

---

## Day 1: Datasets, loaders, episode windows

Folder: `d1-datasets-loaders`. Exercise: `w2d1_data.py`.

### Why this matters

Last Friday you sliced batches by hand from one big tensor. Real robot data does not fit that:
it is a list of episodes of different lengths, read from disk, and one training sample is a
**window** inside an episode. PyTorch splits the job in two: a `Dataset` knows how to produce
**one** sample, a `DataLoader` turns samples into shuffled batches.

### Dataset

Any class with two methods.

```python
class Squares(Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):                    # how many samples
        return self.n
    def __getitem__(self, i):             # sample number i; a dict of tensors is the usual choice
        return {"x": torch.tensor(float(i)), "y": torch.tensor(float(i * i))}

ds = Squares(10)
len(ds), ds[3]                            # 10, {'x': tensor(3.), 'y': tensor(9.)}
```

### DataLoader

```python
g = torch.Generator().manual_seed(0)
loader = DataLoader(ds, batch_size=4, shuffle=True, generator=g, drop_last=False)
for batch in loader:
    print(batch["x"].shape)               # (4,), (4,), (2,): the last batch is smaller
```

| Argument | Meaning |
|---|---|
| `batch_size` | samples per batch |
| `shuffle=True` | a new random order every epoch |
| `generator=` | a seeded random generator: the order is then the same on every run |
| `drop_last` | drop the last, smaller batch |
| `collate_fn` | the function that turns a list of samples into one batch |
| `sampler=` | replaces `shuffle`: an object that decides which indices are drawn |
| `num_workers` | background processes that load samples; 0 = in the main process |

Without `generator=`, two runs see different orders and you cannot reproduce a result.

### Collate

The loader calls `dataset[i]` for each index of the batch and gets a **list** of samples. The
collate function merges that list into one batch. For a list of dicts: for each key, `stack`
the tensors on a new first axis; plain Python numbers become one tensor.

```python
samples = [ds[0], ds[1], ds[2]]
{"x": torch.stack([s["x"] for s in samples])}     # the idea, for one key
```

The default collate does this for you and fails when the tensors of a key have different
shapes. Writing it once by hand tells you what a batch is.

### Windows over episodes

A sample for a chunked policy is: the observation at time `t`, and the next `H` actions
`a[t : t + H]`. Every frame of every episode is the start of one window. So the dataset length
is the total number of frames.

**From a global index to (episode, t).** Sample 0 to `len(ep0) − 1` are in episode 0, the next
ones in episode 1, and so on. Keep the cumulative lengths and search in them:

```python
lengths = torch.tensor([3, 5, 2])
ends = lengths.cumsum(0)                               # tensor([3, 8, 10]): one past the last index of each episode
i = 6
e = int(torch.searchsorted(ends, i, right=True))       # 1: the first episode whose end is > i
t = i - (int(ends[e - 1]) if e > 0 else 0)             # 3: position inside that episode
```

A Python list of all `(episode, t)` pairs works too, and costs memory on a dataset of millions
of frames. The cumulative lengths are one number per episode.

**The end of an episode.** For the last frames, `t + H` runs past the end. The window must not
continue into the **next** episode: that would be an action from another scene. Instead the
window is padded to length `H` and comes with a mask, `True` on real steps. LeRobot does exactly
this (`action_is_pad`, with the opposite sign). What to pad with is a choice; this lab repeats
the last action. The loss ignores the padded steps (week 1, day 2).

Slicing past the end of a tensor is not an error in PyTorch: `a[8:12]` on a tensor of length 10
returns 2 elements. Count how many you got, and pad the rest.

### Balanced sampling

Your bowls dataset has two tasks. If one has three times more frames, a uniform sampler shows it
three times more often. A `WeightedRandomSampler` draws each sample with a probability
proportional to a weight you give.

```python
from torch.utils.data import WeightedRandomSampler
labels = torch.tensor([0, 0, 0, 1])
counts = torch.bincount(labels)                        # tensor([3, 1]): samples per class
counts[labels]                                         # tensor([3, 3, 3, 1]): each sample's class size
```

To make every class equally likely, a sample's weight must shrink as its class grows.

### Traps of the day

- A window that reads actions from the next episode. No error, wrong data.
- No `generator=`: the run cannot be reproduced.
- An index list with one Python tuple per frame.
- Padded steps counted in the loss.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `EpisodeWindowDataset` | one sample per frame: `obs (Do,)`, `actions (H, Da)`, `action_mask (H,)`, `episode` | windows over episodes |
| `collate_windows` | list of samples → batch dict, by hand | collate |
| `make_loader` | seeded, reproducible shuffle | DataLoader |
| `task_balanced_weights` | per-sample weights, every task the same total weight, sum 1 | balanced sampling |

Run: `uv run pytest weeks/w2-training-real-data/d1-datasets-loaders -k "not review"`

---

## Day 2: Training that does not blow up

Folder: `d2-training-stability`. Exercise: `w2d2_stability.py`.

### Why this matters

A loop that is correct can still diverge: the loss jumps, then becomes `NaN`. Four habits
prevent most of it or locate it fast. Each is a few lines, and each is a common interview item.

### Habit 1: know the loss at step 0

A fresh network outputs values near zero, and for classification, nearly equal scores.

- **Regression**, predictions ≈ 0: the MSE is the mean of `y²`. With normalised targets, about 1.
- **Classification** over `K` classes, equal probabilities `1/K`: the cross-entropy is
  `−ln(1/K) = ln K`. For 10 classes, 2.30.

A first loss far from that value is a bug that exists before any learning: targets not
normalised, wrong labels, a bad initialisation.

**Cross-entropy in one paragraph.** A classifier outputs `K` raw scores, the **logits**.
**Softmax** turns them into probabilities: exponentiate, divide by the sum. The loss is minus
the log of the probability given to the correct class. Logits (2, 1, 0) give probabilities
(0.665, 0.245, 0.090); if class 0 is correct the loss is `−ln 0.665 = 0.41`, if class 2 is
correct it is 2.41. `F.cross_entropy(logits, target)` takes the raw logits and integer targets.

### Habit 2: warm up, then decay

Adam divides each step by a running estimate of the gradient's size. In the first steps that
estimate is built from very few gradients and is unreliable; a full learning rate there can
wreck the weights. So the learning rate rises linearly from 0 to its peak over `W` steps
(**warm-up**). Near the end, smaller steps settle into the minimum: the rate follows half a
cosine down to a floor (**cosine decay**).

```
step < W :  lr = peak · step / W
after    :  p  = (step − W) / (total − W)                    progress from 0 to 1
            lr = final + (peak − final) · ½ · (1 + cos(π · p))
```

Check the three boundary values by hand: step 0, step `W`, step `total`. A schedule is counted
in **steps** or in **epochs**; using one where the other is expected is a silent bug.

In a real script the function is wrapped so the optimiser uses it:

```python
net = nn.Linear(4, 2)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)                       # lr here is the peak
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, step / 100))   # factor × peak
# each step: opt.step(); sched.step()
```

### Habit 3: clip the global gradient norm

One bad batch (a teleoperation glitch, a loss spike) can produce a huge gradient and throw the
weights far away. Clipping bounds the size of the step.

The **global norm** treats all gradients of the model as one long vector:
`g_norm = sqrt(sum over every parameter of sum(grad²))`. If `g_norm > max_norm`, every gradient
is multiplied by `max_norm / g_norm`. The direction is kept, only the length changes.

It can only work **after** `backward()` (before that there are no gradients) and **before**
`step()` (after that the weights have moved).

```python
loss = net(torch.randn(8, 4)).pow(2).mean()
opt.zero_grad()
loss.backward()
g_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)   # returns the norm BEFORE clipping
opt.step()
```

Log `g_norm` at every step. A norm that grows over time is the earliest warning you get.
Gradients are changed inside `torch.no_grad()` or through `p.grad` directly; a parameter that
took no part in the loss has `p.grad is None`.

### Habit 4: find the first NaN, not the last

`NaN` spreads: one bad value in an early layer makes everything after it `NaN`, including the
loss. The loss tells you that something broke, not where. A **forward hook** is a function
PyTorch calls each time a module produces its output.

```python
seen = []
def hook(module, inputs, output):
    seen.append((type(module).__name__, bool(torch.isfinite(output).all())))

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
handles = [m.register_forward_hook(hook) for m in model]       # one handle per hook
model(torch.randn(3, 4))
for h in handles:
    h.remove()                                                 # always remove them
seen                                                           # in execution order
```

`model.named_modules()` yields `(name, module)` for every module at every depth, the container
included. A **leaf** module has no children: `len(list(m.children())) == 0`. Hooks on leaves
give the finest location. Hooks that are never removed keep firing on every later call.

`torch.isfinite(t)` is `False` on `nan`, `inf` and `-inf`.

### Traps of the day

- Clipping before `backward()`: it clips last step's gradients, or nothing.
- A schedule stepped per epoch when it was defined in steps.
- Hooks left attached.
- Reading the norm after clipping and concluding all is calm.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `initial_mse_if_zero`, `initial_ce` | the expected first loss | habit 1 |
| `warmup_cosine_lr` | the schedule as a plain function | habit 2 |
| `clip_grad_norm` | global-norm clipping without `torch.nn.utils` | habit 3 |
| `first_nonfinite_module` | name of the first leaf module with a non-finite output | habit 4 |

Run: `uv run pytest weeks/w2-training-real-data/d2-training-stability -k "not review"`

---

## Day 3: CNN encoders for policies

Folder: `d3-cnn-encoders`. Exercise: `w2d3_cnn.py`.

### Why this matters

A policy needs a short vector that says what is in the image and **where**. An MLP on raw pixels
does not scale: a 224×224 colour image is 150 528 numbers, and an MLP treats the same object at
two positions as two unrelated inputs. A convolutional network (CNN) is the classic image
encoder of ACT, Diffusion Policy and robomimic.

### Images as tensors

PyTorch wants `(B, C, H, W)`: batch, channels, height, width, values usually in [0, 1]. Image
files and cameras give `(H, W, C)` with values 0–255. Going from one to the other is a
**permute** (week 1, day 1). A `reshape` gives the right shape and a scrambled image.

### Convolution

A small grid of weights, the **kernel** (say 3×3), slides over the image. At each position it
multiplies the patch under it by the kernel and sums: one output number. Sliding over all
positions gives an output image, a **feature map**. The same kernel is used everywhere, so a
pattern is detected wherever it appears.

With `C` input channels and `C_out` kernels, the weight has shape `(C_out, C, k, k)` and the
output has `C_out` channels.

- **stride** `s`: the kernel moves `s` pixels at a time; stride 2 halves the resolution.
- **padding** `p`: zeros added around the border so the kernel can reach the edges.

Output size: `H_out = floor((H + 2p − k) / s) + 1`, same for the width.

```python
conv = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
conv(torch.randn(8, 3, 64, 64)).shape        # (8, 16, 32, 32): (64 + 4 − 5) // 2 + 1 = 32
conv.weight.shape                            # (16, 3, 5, 5)
```

### A convolution is a matrix product

`F.unfold` cuts every `k×k` patch out of the image and lays each one flat as a column.

```python
x = torch.randn(2, 3, 8, 8)
cols = F.unfold(x, kernel_size=3, stride=1, padding=0)
cols.shape                                   # (2, 27, 36): 3·3·3 numbers per patch, 6·6 patches
```

`(B, C·k·k, L)` where `L` is the number of patch positions. One output number is one kernel
(flattened to `C·k·k`) times one column. All kernels against all columns is one matrix product;
the result has `L` in its last axis, which you fold back into `(H_out, W_out)` with the formula
above. Then the bias, one per output channel, is added by broadcasting.

### Receptive field

The **receptive field** of an output unit is the number of input pixels it depends on. It grows
with each layer, and faster after a stride. Walk through the layers with two numbers, the
receptive field `r` and the jump `j` (how many input pixels one step in the current feature map
is worth):

```
r = 1, j = 1
for each layer (k, s):   r = r + (k − 1) · j ;   j = j · s
```

Two 3×3 stride-1 layers: `r = 1 → 3 → 5`. If the gripper is 40 pixels wide and the receptive
field is 15, no single feature sees the whole gripper.

### From a feature map to a vector: spatial softmax

After the convolutions you have `(B, C, H, W)` and need `(B, something)`.

- **Global average pooling** averages each channel over all positions: `(B, C)`. It keeps *what*
  is there and loses *where*. Fine for classification, bad for control.
- **Spatial softmax** keeps *where*. For each channel: softmax over all `H·W` positions, which
  turns the map into a probability map; then the expected `x` and expected `y` under it. Each
  channel becomes one learned keypoint. Output `(B, 2C)`.

The building blocks:

```python
feat = torch.randn(2, 4, 5, 7)                                    # (B, C, H, W)
p = F.softmax(feat.flatten(2), dim=-1)                            # (B, C, H·W), each row sums to 1
xs = torch.linspace(-1, 1, 7)                                     # x coordinate of each column
ys = torch.linspace(-1, 1, 5)                                     # y coordinate of each row
gy, gx = torch.meshgrid(ys, xs, indexing="ij")                    # both (H, W)
gx.flatten().shape                                                # (H·W,): x of every position, same order as p
```

An expected value is a sum of `probability × value`. Coordinates run in [−1, 1], `x` along the
width and `y` along the height. Read the docstring for the output order.

### Normalisation layers in a policy

**BatchNorm** normalises each channel with the mean and variance of the **batch** during
training, and with running averages at test time. On the arm the batch is one image, and
training batches often hold frames of the same episode. Train and test then behave differently.

**GroupNorm** splits the channels into groups and normalises each group **within one image**.
No dependence on the batch: same behaviour in training and on the robot. Diffusion Policy
replaces every BatchNorm of its ResNet with GroupNorm.

```python
nn.GroupNorm(num_groups=8, num_channels=32)(torch.randn(2, 32, 16, 16)).shape    # unchanged shape
```

### Random shift

The cheapest augmentation that works for image policies (from DrQ): pad the image by a few
pixels by repeating its border, then crop back to the original size at a random offset,
different for each image of the batch. The policy learns that a camera moved by a few pixels is
the same scene.

```python
img = torch.randn(4, 3, 32, 32)
padded = F.pad(img, (4, 4, 4, 4), mode="replicate")       # (left, right, top, bottom) → (4, 3, 40, 40)
dx = torch.randint(0, 9, (4,))                            # one offset per image, 0..2·pad
```

Augmentation is for **training** batches only. An evaluation with augmentation on measures a
different, noisier thing.

### Traps of the day

- `(H, W, C)` reshaped to `(C, H, W)` instead of permuted.
- Off by one in the output size: use the formula, check with a real `nn.Conv2d`.
- Coordinates and probabilities flattened in different orders.
- BatchNorm in a policy.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `conv2d_unfold` | same numbers as `F.conv2d`, from `F.unfold` and a matmul | a convolution is a matrix product |
| `receptive_field` | from a list of `(kernel, stride)` | receptive field |
| `SpatialSoftmax` | `(B, C, H, W)` → `(B, 2C)` keypoints | spatial softmax |
| `CNNEncoder` | 3 conv layers with GroupNorm and ReLU, spatial softmax, a Linear | all of the above |
| `random_shift` | replicate-pad then random crop, per sample | random shift |

Run: `uv run pytest weeks/w2-training-real-data/d3-cnn-encoders -k "not review"`

---

## Day 4: Your real SO-101 data, by hand

Folder: `d4-real-so101-data`. Exercise: `w2d4_lerobot.py`. Provided: `fake_lerobot.py` (a fake
dataset in the real layout, used by the tests), `download_bowls.py`, `inspect_real.py`.

### Why this matters

A LeRobot dataset is tables, videos and one JSON file. Nothing hidden. Read it yourself once and
you will never wonder what the loader did. Today's data is yours:
`ravediamond/so101_bowls_all_trimmed`, 182 episodes, 61 462 frames at 30 fps, 2 cameras.

### What is on disk

```
<root>/meta/info.json          fps, features, shapes, counts, version
<root>/data/**/*.parquet       one row per frame: state, action, indices, timestamp
<root>/videos/**/*.mp4         the camera streams
```

**Parquet** is a table file format, column by column, compressed. Dataset format v3 stores many
episodes per file (`file-000.parquet`, …); v2 stores one file per episode
(`episode_000000.parquet`, …). Code that reads every `.parquet` under `data/` handles both.

Every row has:

| Column | Meaning |
|---|---|
| `observation.state` | follower joint positions, 6 numbers |
| `action` | leader command, 6 numbers |
| `episode_index` | which episode |
| `frame_index` | position inside the episode, from 0 |
| `timestamp` | seconds since the start of the episode |
| `index` | global frame number over the whole dataset |
| `task_index` | which task |

### The pandas you need

A `DataFrame` is a table with named columns.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.DataFrame({"episode_index": [1, 0, 0], "frame_index": [0, 1, 0],
                   "action": [np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32)]})
df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)   # order rows, renumber them
np.stack(df["action"].to_numpy()).shape           # (3, 2): a column of arrays → one 2-D array
for ep, group in df.groupby("episode_index"):     # one sub-table per episode, in key order
    print(ep, len(group))
```

- `pd.read_parquet(path)` reads one file into a DataFrame.
- `pd.concat([df1, df2], ignore_index=True)` stacks tables.
- `sorted(Path(root, "data").rglob("*.parquet"))` lists every parquet file at any depth.
- `json.loads(Path(root, "meta", "info.json").read_text())` reads the metadata.

A column that holds one array per row has dtype `object`. `np.stack` turns it into a real
`(N, 6)` array.

**Never rely on the order of rows.** Files may be listed in a different order on another
machine, and rows inside a file are not guaranteed to be in time order. Sort explicitly by the
index columns. That is today's review bug.

### What real data does that toy data did not

**Gaps.** At 30 fps two consecutive frames are `1/30 = 0.033 s` apart. A dropped frame shows up
as a larger step in `timestamp`. `np.diff(timestamps)` gives the steps; count those clearly
above `1/fps`.

**Non-finite values.** A `NaN` in one frame poisons a whole batch. `np.isfinite(a).all()`.

**Idle starts.** The operator waits before moving, so the first frames all carry the same
action. A policy trained on them learns that "do nothing" is a good answer at the start. You
trimmed them from this dataset; the check finds how many idle frames an episode starts with.

**Lag.** `action` is what the leader arm commanded. `observation.state` is where the follower
actually is. The follower trails the command by a few frames. To measure it, shift one signal
against the other and keep the shift with the smallest error:

```python
a = np.arange(10.0)                      # a command
s = np.concatenate([[0.0, 0.0], a[:-2]]) # a state that trails it by 2 frames
k = 2
np.abs(s[k:] - a[:len(a) - k]).mean()    # 0.0 at the true lag; larger for other k
```

That lag is why "hold position" is not a zero-error baseline, and one reason chunked policies help.

### After the tests pass, on your Mac

```bash
cd weeks/w2-training-real-data/d4-real-so101-data
uv run python download_bowls.py
uv run python inspect_real.py
```

Compare the output with the numbers above and write it in `PROGRESS.md`.

### Traps of the day

- Trusting file order or row order.
- Comparing float timestamps with `==`; use a threshold.
- Stacking an `object` column without `np.stack`.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `read_info` | parse `meta/info.json` | what is on disk |
| `load_frames` | every parquet file, concatenated, sorted by the global `index` | the pandas you need |
| `to_episodes` | one dict per episode with `(T, 6)` float32 arrays | the pandas you need |
| `check_episode` | gaps, non-finite values, idle start | what real data does |
| `action_state_lag` | the number of frames the follower trails the command | lag |

Run: `uv run pytest weeks/w2-training-real-data/d4-real-so101-data -k "not review"`

---

## Day 5: A pixel + state policy

Folder: `d5-pixel-policy-real`. Exercise: `w2d5_pixel_policy.py`. Provided: `dot_world.py` (toy
pixel world), `cnn_ref.py` (a reference encoder, if you prefer it to your day-3 one),
`extract_frames.py` and `train_real.py` (for your real data).

### Why this matters

The joint state says where the arm is. The image says where the bowl is **today**. A policy that
ignores the image can only learn the average trajectory. Today you build a policy that uses
both, and you **prove** the image branch is used.

### The architecture

```
img   (B, 3, S, S) → random shift (training only) → CNN → (B, 64) ┐
state (B, Ds)      → normalise → Linear            → (B, 64) ┴ concat → MLP → (B, H·Da) → (B, H, Da)
```

- **Fusion by concatenation**: each input is encoded to a vector, the vectors are joined with
  `torch.cat(..., dim=-1)`, an MLP reads the result.
- **Chunk output**: the last layer produces `H·Da` numbers, reshaped to `(B, H, Da)`: the next
  `H` actions.
- **Statistics in buffers**, raw in and raw out, loss in normalised action space: exactly week 1,
  day 5, with one more axis. The action statistics are per action dimension, computed over all
  samples **and** all chunk steps that are real.

### The masked chunk loss

Prediction and target are `(N, H, D)`, the mask is `(N, H)`. It is the masked mean of week 1,
day 2: squared error per entry, zero out padded steps, divide by the number of real **entries**.
One real step holds `D` entries, so the count is the number of real steps times `D`.

### The ablation twin

In `dot_world.py` the target is visible **only** in the image (a bright dot); the state is the
tip position. A state-only policy cannot know where to go. So you train the same policy twice:
once with the image, once with `use_image=False` (no CNN, same head). If the full policy does
not beat its twin by a wide margin, the image branch is broken or ignored, whatever the loss
curve looks like.

This is the general method for the question "does my model use input X?": remove X, retrain,
compare on the same validation set. The test asks for less than 30% of the twin's validation MSE.

### Training and evaluation, the parts that differ from week 1

- Augmentation (`random_shift`) on training batches only.
- `evaluate` in raw units, `model.eval()`, no graph, and in chunks of a few hundred samples so
  the images fit in memory.
- `train_policy` returns the policy in eval mode.

### After the tests pass, on your Mac (about 20 min, once)

```bash
cd weeks/w2-training-real-data/d5-pixel-policy-real
uv sync --extra video
uv run python ../d4-real-so101-data/download_bowls.py --videos
uv run python extract_frames.py ~/data/so101_bowls_all_trimmed --camera front_top_camera --size 64
uv run python train_real.py ~/data/so101_bowls_all_trimmed --episodes 60
```

`extract_frames.py` prints its decoding speed in frames per second: that number is why loaders
cache decoded frames. Write the three validation numbers in `PROGRESS.md`.

### Traps of the day

- Augmentation left on at evaluation.
- Loss divided by `N·H·D` instead of the number of real entries.
- A "state-only" twin that still receives the image through some path.
- Statistics computed over padded steps.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `masked_chunk_mse` | mean over real `(step, dim)` entries | the masked chunk loss |
| `PixelStatePolicy` | the architecture above, with a `use_image` switch | the architecture, the ablation twin |
| `train_policy` | stats, Adam, masked MSE in normalised space, augmentation in training | training and evaluation |
| `evaluate` | masked chunk MSE in raw units | training and evaluation |

Run: `uv run pytest weeks/w2-training-real-data/d5-pixel-policy-real -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w2` (two pointers, sliding window), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 2 (how LeRobot builds a window).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 2 when you can do these without notes:

- [ ] write a `Dataset` of windows over episodes, with padding and a mask, and a seeded loader
- [ ] give the step-0 loss for regression and for K-class classification
- [ ] write warm-up + cosine as a function, and global-norm clipping by hand
- [ ] compute a conv output size and a receptive field; explain spatial softmax and why GroupNorm
- [ ] name five checks on a teleoperation dataset before training on it
- [ ] design an ablation that proves a model uses one of its inputs

## Optional depth (not needed for the labs)

- Data, optimisers and debugging tools, longer: `reference/journal/book/00p2-pytorch-for-the-coding-round.md` (P2.7, P2.8, P2.14)
- Your own data pipeline: `reference/journal/book/10b-the-data-pipeline.md`
- Previous week: [`../w1-pytorch-basics/README.md`](../w1-pytorch-basics/README.md) · Next: [`../w3-transformers-gpt2/README.md`](../w3-transformers-gpt2/README.md)
