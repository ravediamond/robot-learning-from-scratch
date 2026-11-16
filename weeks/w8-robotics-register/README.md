# Week 8: Robotics basics and your own register

**Everything you need for this week is on this page.** You need week 1 (tensors, `state_dict`)
and week 6 (Wilson, AUROC, bootstrap). Days 4 and 5 read files that are already in this repo
under `reference/`; the lesson says exactly which ones.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [The idea of the week](#the-idea-of-the-week).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-kinematics),
   [Day 2](#day-2-cameras-and-calibration), [Day 3](#day-3-control-latency-chunk-boundaries),
   [Day 4](#day-4-re-derive-your-own-numbers), [Day 5](#day-5-checkpoint-forensics).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w8dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w8dN_*.py` | tests | run, never edit |
| `review_w8dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w8-robotics-register/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w8-robotics-register/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w8-robotics-register/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w8-robotics-register/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-kinematics` | rotations, transforms, forward and inverse kinematics | FK, Jacobian and IK for an SO-101-like arm | one rotation matrix |
| Tue | `d2-cameras-calibration` | the pinhole camera, hand-eye calibration | projection, back-projection, a rigid fit | what you divide by |
| Wed | `d3-control-latency` | PID, delay, chunk boundaries | a PID, a chunk-switching simulator | one term of the PID |
| Thu | `d4-rederive-your-numbers` | auditing published numbers | recompute your own intervals and AUROCs from raw files | the layout of a table |
| Fri | `d5-checkpoint-forensics` | how checkpoint loads fail silently | a load report, key remapping, a safe load | a return value nobody reads |

## The idea of the week

Two halves.

**Days 1–3: the robot itself.** A learned policy sits on top of classical robotics: where the
gripper is given the joint angles, where a pixel is in the world, how a command becomes motion
and how late it arrives. A robotics interview checks that you know this layer, and most
real-robot bugs live in it.

**Days 4–5: your own evidence.** You have a register of results and a paper. This week you
recompute published numbers from raw files with code you wrote in this course, and you turn the
checkpoint-loading bug you lived through into a tool. The rule for an interview: **a number you
cannot re-derive yourself does not get said out loud.**

All code on this page assumes:

```python
import math
import torch
import torch.nn as nn
```

Days 1 and 2 use `float64` everywhere: geometry accumulates rounding error, and the tests compare
to tight tolerances. `torch.tensor(..., dtype=torch.float64)`.

---

## Day 1: Kinematics

Folder: `d1-kinematics`. Exercise: `w8d1_kinematics.py`.

### Frames and rotations

A **frame** is a coordinate system attached to something: the base of the robot, a link, the
gripper, a camera. The same point has different coordinates in different frames. Robotics is
mostly bookkeeping of frames.

A **rotation matrix** `R` (3×3) turns coordinates in one frame into coordinates in another. The
three elementary rotations, by angle `θ` about each axis, with `c = cos θ`, `s = sin θ`:

```
Rz(θ) = [[c, −s, 0],      Ry(θ) = [[ c, 0, s],      Rx(θ) = [[1, 0,  0],
         [s,  c, 0],               [ 0, 1, 0],               [0, c, −s],
         [0,  0, 1]]               [−s, 0, c]]               [0, s,  c]]
```

Note where the minus sign sits in `Ry`: bottom left, the opposite corner from `Rz` and `Rx`.
It follows from the right-hand rule and it is the classic mistake.

Two checks that hold for every rotation matrix: `R @ R.T` is the identity, and `det(R) = 1`.
A sanity check for the signs: a rotation by +90° about z sends the x axis to the y axis.

To build a matrix from a 0-d tensor angle and keep it differentiable, assemble it with
`torch.stack` from tensors, not with `torch.tensor([...])` on Python floats.

```python
t = torch.tensor(0.3, dtype=torch.float64)
c, s = torch.cos(t), torch.sin(t)
zero, one = torch.zeros_like(t), torch.ones_like(t)
torch.stack([torch.stack([c, -s]), torch.stack([s, c])])      # a 2-D rotation, still a function of t
```

### Homogeneous transforms

A **pose** is a rotation and a translation `p`. Both fit in one 4×4 matrix:

```
T = [[ R, p ],
     [ 0, 1 ]]
```

A point `(x, y, z)` is written `(x, y, z, 1)` so that `T @ point` rotates and translates in one
product. Poses **chain by matrix product**:

```
T_world_tool = T_world_base @ T_base_link1 @ T_link1_link2 @ …
```

Read the subscripts like dominoes: the inner names must match. The order matters: a rotation
followed by a translation is not the translation followed by the rotation.

### Forward kinematics (FK)

**Given the joint angles, where is the gripper tip?** Walk from the base to the tip, multiplying
one transform per joint (a rotation by the joint angle) and one per link (a fixed translation).

The arm of this lab, close to the SO-101, lengths in metres:

```
base:      Rz(q0) shoulder pan,   then up 0.10 along z
shoulder:  Ry(q1) lift,           then 0.116 along x      upper arm
elbow:     Ry(q2) flex,           then 0.135 along x      forearm
wrist:     Ry(q3) flex, Rx(q4) roll, then 0.10 along x    to the gripper tip
```

With all angles at 0 the arm lies straight along x: the tip is at `(0.351, 0, 0.10)`, which is
`0.116 + 0.135 + 0.10` forward and `0.10` up. Check that first.

### The Jacobian

The **Jacobian** `J` says how the tip moves when each joint moves a little: entry `(i, j)` is
`∂ tip_i / ∂ q_j`. Position only, 5 joints: shape `(3, 5)`. Column `j` is the velocity of the tip
per unit of velocity of joint `j`.

Numerically it is the finite difference of week 1, day 3, one joint at a time: move joint `j`
by `+ε` and `−ε`, take FK, divide the difference of the tip positions by `2ε`.

### Inverse kinematics (IK)

**Given a target position, which joint angles reach it?** No general formula; it is solved by
iteration. For a small move, `Δtip ≈ J Δq`. You want `Δtip = e`, the error `target − tip(q)`.

Five joints for three coordinates: `J` is not square, and many `Δq` give the same `Δtip`.
**Damped least squares** picks a small one and stays stable:

```
Δq = Jᵀ ( J Jᵀ + λ² I )⁻¹ e
q  ← q + Δq          repeat until ||e|| is small
```

`J Jᵀ` is 3×3. Solve the system with `torch.linalg.solve`, do not invert. The **damping** `λ`
matters near a **singularity**: a pose where the arm loses a direction of motion, such as fully
stretched (it cannot move further outward). There `J Jᵀ` is nearly non-invertible and the
undamped step explodes; with `λ` the step stays bounded, at the price of slower convergence.

A target out of reach has no solution: the iteration stops with a non-zero error, which the
function returns.

### Traps of the day

- The sign pattern of `Ry`.
- Transforms multiplied in the wrong order.
- float32: the tests fail by rounding.
- An undamped inverse at full stretch.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `rot_x`, `rot_y`, `rot_z` | the three elementary rotations | frames and rotations |
| `transform` | 4×4 from a rotation and/or a translation | homogeneous transforms |
| `fk` | joint angles → pose of the tip | forward kinematics |
| `jacobian` | `(3, 5)` by central differences | the Jacobian |
| `ik_dls` | damped least squares, returns `q` and the final error | inverse kinematics |

Run: `uv run pytest weeks/w8-robotics-register/d1-kinematics -k "not review"`

---

## Day 2: Cameras and calibration

Folder: `d2-cameras-calibration`. Exercise: `w8d2_camera.py`.

### The pinhole camera

A camera turns a 3-D point into a pixel in two steps.

**Extrinsics: world to camera frame.** A 4×4 transform, `p_cam = T_cam_world @ [p_world, 1]`.
The camera frame (OpenCV convention): x to the right, y **down**, z **forward** along the
optical axis.

**Intrinsics: camera frame to pixel.** A point at `(X, Y, Z)` in the camera frame lands at

```
u = fx · X / Z + cx          column
v = fy · Y / Z + cy          row
```

`fx`, `fy` are the focal lengths in pixels; `(cx, cy)` is the pixel where the optical axis hits
the image, near the centre. They are grouped in the matrix
`K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.

The division by `Z` is perspective: twice as far, half as big. `Z` is the **depth**, the
coordinate along the optical axis. It is not the distance from the camera to the point: for a
point away from the centre of the image the distance is larger. Dividing by the distance bends
the image. That is today's review bug.

### Back-projection

A pixel alone is a ray: every point along it projects to the same pixel. With the depth `Z` of
that pixel (from a depth camera) the point is recovered by inverting the two steps:

```
X = (u − cx) · Z / fx        Y = (v − cy) · Z / fy        p_cam = (X, Y, Z)
p_world = T_world_cam @ [p_cam, 1]            T_world_cam is the inverse of T_cam_world
```

`torch.linalg.inv(T)` inverts a transform. Projecting a point and back-projecting the result
with its depth must return the point: that round trip is the test.

For many points at once, append the column of ones with `torch.cat`, multiply all of them in
one product (`points_h @ T.T` for points stored as rows), and drop the last column.

### Hand-eye calibration as a rigid fit

The policy sees the world through the camera and acts in the robot's frame. The transform
between the two must be measured. Move the gripper to `N` poses. For each, you know the tip
position in the **robot** frame (FK, yesterday) and you measure it in the **camera** frame (a
marker and depth). Find the rotation `R` and translation `t` that best map one set onto the other:

```
B ≈ R A + t            A: robot-frame points (N, 3),  B: camera-frame points (N, 3)
```

**The Kabsch algorithm** solves it in closed form:

1. Centre both sets: subtract each set's mean. The translation is gone; only a rotation remains.
2. `H = A_cᵀ B_c`, a 3×3 matrix.
3. Singular value decomposition: `H = U S Vᵀ` (`torch.linalg.svd` returns `U`, `S`, `Vᵀ`).
4. `R = V · diag(1, 1, d) · Uᵀ`, with `d = sign(det(V Uᵀ))`.
5. `t = mean(B) − R · mean(A)`.

The `d` matters. Without it the best fit can be a **reflection** (a mirror image, determinant
−1), which is not a physical rotation. It happens with noisy or nearly flat sets of points.

**The fit error** (RMSE, the root of the mean squared distance between the mapped `A` and `B`) is
the first number to look at after a calibration: a few millimetres is good; centimetres mean a
bad marker detection or a wrong pairing of points.

### Traps of the day

- Dividing by the distance instead of the depth.
- `u` and `v`, or `x` and `y`, swapped.
- A reflection returned as a rotation.
- Points as rows multiplied as if they were columns: `A @ R.T`, not `R @ A`.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `intrinsics` | the matrix `K` | the pinhole camera |
| `project` | world points → pixels and depths | the pinhole camera |
| `backproject` | pixels and depths → world points | back-projection |
| `kabsch` | `R` (determinant +1) and `t` with `B ≈ A @ R.T + t` | hand-eye calibration |
| `rigid_rmse` | the fit error | hand-eye calibration |

Run: `uv run pytest weeks/w8-robotics-register/d2-cameras-calibration -k "not review"`

---

## Day 3: Control, latency, chunk boundaries

Folder: `d3-control-latency`. Exercise: `w8d3_control.py`.

### The control loop

A policy outputs a target; a **controller** turns the gap between target and measurement into a
motor command, many times per second. The SO-101's servos run such a loop inside.

**PID**, the standard controller, in one screen:

```
e = target − measured
u = kp · e  +  ki · Σ(e · dt)  +  kd · (e − e_prev) / dt
```

| Term | Reacts to | Effect |
|---|---|---|
| **P** proportional | the error now | pushes toward the target; alone, it leaves a steady error against a constant disturbance (gravity, friction) |
| **I** integral | the error accumulated over time | removes that steady error; can overshoot and "wind up" |
| **D** derivative | how fast the error changes | damps the motion |

The integral **multiplies** by `dt` (a sum over time); the derivative **divides** by `dt` (a
rate). Getting one of the two wrong changes the gains by a factor of `dt` or `dt²` and is
today's review bug.

A controller object keeps state between calls: the running integral and the previous error.

### Delay

Every real loop has delay: sensing, computing, communication. A command computed now acts
later. The controller then pushes against an error that is already out of date. With low gains
nothing bad happens; with high gains it overcorrects, then overcorrects the overcorrection:
**oscillation**. The more delay, the lower the gain you can afford.

In the simulation, a delay of `d` steps means the plant receives at step `t` the command that
was computed at step `t − d`, and zero before any command exists. A list of past commands, or
a queue, holds them.

### Chunks with inference delay

A policy looks at the world at time `t` and plans a chunk of `H` future commands. Inference
takes `d` control steps. The chunk is ready at `t + d`, and its first `d` commands were meant
for moments that are already in the past. Three ways to switch to the new chunk:

| Mode | At time `t + d`, execute | Result |
|---|---|---|
| **naive** | `chunk[0]` | every command is `d` steps late: lag, and a jump at the switch |
| **shift** | `chunk[d]` | no lag; still a jump if the new chunk disagrees with the old one |
| **blend** | a cross-fade from the old chunk's remaining commands to the new chunk over the first `m` steps | smooth boundary |

Two chunks disagree for a good reason: a generative policy samples, and two samples can be two
different ways of doing the task. The jump at the boundary is the **boundary problem**, and
real-time chunking (RTC) is the family of methods that solves it; the blend here is its
simplest form. Your edge-inference paper is about exactly this time budget.

Indexing is the whole difficulty. For a chunk requested at time `t`, with entry `i` meant for
time `t + 1 + i`, work out on paper which entry corresponds to "now" in each mode before coding.
The docstring of `run_chunked` gives the exact rule; follow it literally.

### Traps of the day

- `dt` on the wrong side in the I or D term.
- The derivative on the first step, when there is no previous error.
- An off-by-one between "requested at `t`" and "meant for `t + 1`".
- A blend whose weights do not go from old to new.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `PID` | the controller, with its state | the control loop |
| `simulate_plant` | a plant with a delayed command and a disturbance | delay |
| `run_chunked` | naive, shift and blend switching | chunks with inference delay |

Run: `uv run pytest weeks/w8-robotics-register/d3-control-latency -k "not review"`

---

## Day 4: Re-derive your own numbers

Folder: `d4-rederive-your-numbers`. Exercise: `w8d4_rederive.py`.

### The files

Real files from your own work, already in this repo:

| File | Content |
|---|---|
| `reference/journal/papers/failure-monitors/results/counts_ci.json` | counts `k` out of `n` and the Wilson intervals your failure-monitors paper reports |
| `reference/journal/papers/jepa-world-model/gaps/anchor_predictions.csv` | per-episode monitor scores: three features × five seeds, on your labelled bowls rollouts |
| `reference/journal/papers/jepa-world-model/gaps/anchor_ci.json` | the AUROCs and intervals you published from those scores |

Open each one and look at it before writing any code.

### What "re-derive" means

1. **Parse the raw file yourself.**
2. **Recompute with code you wrote in this course.** Not a copy of the original script: an
   independent second implementation is the point.
3. **Compare with the stored number at a stated tolerance**, and explain every difference or
   flag it as an error.

Legitimate differences have a name: rounding in the stored value, `z = 1.96` against the exact
`1.959964`, a bootstrap that used another random generator. A bootstrap interval will not match
to the third decimal and should not; a Wilson interval should match to rounding.

### The tools, all from week 6

- **Wilson** interval (week 6, day 2).
- **AUROC** in rank form with average ranks for ties (week 6, day 3). Label 1 = failure.
- **Episode bootstrap** for the interval of an AUROC, skipping resamples with a single class.

### Reading the two formats

```python
import csv, json
from pathlib import Path

rows = [{"episode": "3", "seed": "0", "score": "0.7"}, {"episode": "1", "seed": "0", "score": "0.2"}]
sorted(rows, key=lambda r: int(r["episode"]))      # csv.DictReader yields dicts of STRINGS like these
```

- `json.loads(Path(p).read_text())` gives nested dicts; a dict keeps the order of the file.
- `csv.DictReader(open(p))` yields one dict per row. Every value is a **string**: convert.

### Long tables and the layout trap

The CSV is a **long** table: one row per (episode, feature, seed). You need, per feature, a
matrix of scores of shape `(n_seeds, n_episodes)` and one label per episode.

Never trust the row order of a file to give you that matrix by a reshape. If the file is sorted
seed by seed, a reshape to `(n_episodes, n_seeds)` runs, has a plausible shape, and pairs
scores with the wrong episodes. Place each value by its keys: sort the episode ids, map each id
to a column, each seed to a row, and write `scores[row, column]`. That is today's review bug.

### Two ways to summarise five seeds

- **Mean of seeds**: one AUROC per seed, then their average. It answers "how good is one model,
  typically?".
- **Seed-averaged**: average the five scores of each episode, then one AUROC. It answers "how
  good is the ensemble of five?". It is usually higher, because averaging removes noise.

They are different claims. Know which one each published number is.

### After the tests pass

Write in `PROGRESS.md` the three numbers you would now quote in an interview, each with its
interval, one line each.

### Traps of the day

- Strings compared or sorted as strings: `"10" < "9"`.
- A reshape that assumes a row order.
- A tolerance chosen after seeing the difference.
- Mean-of-seeds reported as if it were the ensemble, or the reverse.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `wilson` | the interval, again, from memory | the tools |
| `audit_counts` | names whose stored bounds differ from yours beyond a tolerance | what "re-derive" means |
| `load_anchor` | the CSV → per feature: episodes, labels, `(n_seeds, E)` scores | long tables |
| `auroc` | rank form, average ranks | the tools |
| `recompute_anchor` | per seed, mean of seeds, seed-averaged | two ways to summarise |
| `bootstrap_auroc_ci` | episode bootstrap interval | the tools |

Run: `uv run pytest weeks/w8-robotics-register/d4-rederive-your-numbers -k "not review" -s`

---

## Day 5: Checkpoint forensics

Folder: `d5-checkpoint-forensics`. Exercise: `w8d5_forensics.py`.

### Why this matters

You lived this one: a model evaluated with part of its weights still at their random
initialisation, and nothing complained. Your audit is in
`reference/journal/papers/jepa-world-model/AUDIT_W1_LOADS.md`. Today you turn it into a function
that makes the mistake impossible.

### How a load fails without an error

A checkpoint is a dict from **names** to tensors. Loading matches names. Everything that can
make names or shapes differ makes a load partial:

- **`strict=False` with the result ignored.** `load_state_dict(sd, strict=False)` loads what
  matches and **returns** the lists of missing and unexpected keys. Code that does not read that
  return value can load nothing at all and continue.
- **Prefixes.** A model saved from inside DDP has every key prefixed with `module.`; a wrapper
  class adds `model.`; LeRobot policies nest sub-modules. `module.encoder.weight` does not match
  `encoder.weight`.
- **Renamed layers** between two versions of the code (`head` became `action_head`).
- **Shape mismatches**: another action dimension, another chunk size.
- **Buffers.** Normalisation statistics that were not saved, or that are re-initialised after
  the load.

```python
net = nn.Linear(3, 2)
sd = {"module.weight": torch.zeros(2, 3), "module.bias": torch.zeros(2)}     # saved from a wrapper
result = net.load_state_dict(sd, strict=False)
result.missing_keys, result.unexpected_keys      # (['weight', 'bias'], ['module.weight', 'module.bias']): NOTHING loaded
```

### The checklist, as code

**1. Count, before loading.** Compare the file's keys and shapes with `model.state_dict()`:

| Category | Meaning |
|---|---|
| expected | keys of the model |
| matched | in both, same shape |
| missing | in the model, not in the file: these stay at their initial values |
| unexpected | in the file, not in the model: these are ignored |
| shape mismatch | in both, different shapes |

Set operations on the keys do it: `set(a) - set(b)`, `set(a) & set(b)`; sort the results so the
report is stable.

**2. Fail loudly** unless the count is exactly what you expect. A safe load raises, with the
report in the message, if anything is missing, unexpected or mismatched; only then loads with
`strict=True`.

**3. Remap names explicitly.** A list of rules `(old_prefix, new_prefix)`; for each key, the
first rule whose old prefix starts the key rewrites it. `key.startswith(old)` and slicing,
`new + key[len(old):]`. An empty old prefix matches every key and adds a prefix.

**4. Verify the content.** Names matching is not proof that values were written. Take a copy of
the model's `state_dict` **before** loading, and after loading count the fraction of tensors
whose values changed. A copy means cloned tensors: `{k: v.clone() for k, v in sd.items()}`. A
plain `dict(sd)` holds the same tensors, which change with the model, so it would always report
"nothing changed". Compare with `torch.equal(a, b)`.

The last and strongest check: outputs on a fixed input equal to outputs recorded when the
checkpoint was saved (week 1, day 5).

### Traps of the day

- An unread `strict=False` result.
- A "before" snapshot that is not a copy.
- A remap rule that matches more keys than intended.
- A fraction changed below 1 explained away instead of investigated. (Some tensors legitimately
  do not change, such as a counter that was 0 in both; know which.)

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `load_report` | the five categories, without loading | the checklist, step 1 |
| `remap_keys` | prefix rules, first match wins | step 3 |
| `safe_load` | raise with the report, or load strictly and return the count | step 2 |
| `fraction_changed` | fraction of tensors whose values changed | step 4 |

Run: `uv run pytest weeks/w8-robotics-register/d5-checkpoint-forensics -k "not review"`
Then read your `AUDIT_W1_LOADS.md` and answer `questions.md`.

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w8` (LRU cache, trie, ring buffer, in-place rotation), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 8 (your RTC controller and LeRobot's action queue).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 8 when you can do these without notes:

- [ ] write `Rz`, `Ry`, `Rx` and a homogeneous transform; compute FK for a 2-link arm by hand
- [ ] explain the Jacobian, damped least squares and what a singularity is
- [ ] write the projection equations and say why depth, not distance
- [ ] explain Kabsch in five steps and why the determinant check
- [ ] write a PID and explain what delay does to it
- [ ] explain naive, shift and blend chunk switching
- [ ] quote three of your own numbers with intervals, re-derived
- [ ] list five ways a checkpoint load fails silently and the four steps that prevent it

## Optional depth (not needed for the labs)

- Robotics fundamentals, longer: `reference/journal/book/10a-robotics-fundamentals.md`
- Real-time execution and the boundary problem: `reference/journal/book/11b-chunking-and-edge-cost-papers.md`
- Timed version for later: `drills/live/L10_quaternions.py`
- Previous week: [`../w7-jepa-world-models/README.md`](../w7-jepa-world-models/README.md) · Next: [`../w9-mock-interviews/README.md`](../w9-mock-interviews/README.md)
