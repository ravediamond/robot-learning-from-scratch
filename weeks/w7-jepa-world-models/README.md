# Week 7: JEPA and world models

**Everything you need for this week is on this page.** You need weeks 1–2 (training, CNNs),
`detach` from week 1 day 3, and AUROC from week 6 day 3. Nothing from other folders is required.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [The idea of the week](#the-idea-of-the-week).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-collapse-and-vicreg),
   [Day 2](#day-2-sigreg-ema-targets-stop-gradient), [Day 3](#day-3-probes-and-baselines),
   [Day 4](#day-4-a-world-model-from-pixels), [Day 5](#day-5-planning-and-surprise).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w7dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w7dN_*.py` | tests | run, never edit |
| `review_w7dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided helpers (days 4 and 5): `pixel_world.py` (the world), `jepa_ref.py` (a reference
SIGReg), `probes_ref.py` (reference probes and copy-last), so those days do not depend on your
own days 2 and 3.

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w7-jepa-world-models/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w7-jepa-world-models/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w7-jepa-world-models/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w7-jepa-world-models/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-collapse-vicreg` | representation collapse; VICReg | the loss, effective rank; watch a collapse | a variance over the wrong axis |
| Tue | `d2-sigreg-ema` | SIGReg; EMA targets and stop-gradient | both families of anti-collapse tools | the two weights of an average swapped |
| Wed | `d3-probes-baselines` | how to know a latent is any good | ridge and MLP probes, R², copy-last | the denominator of R² |
| Thu | `d4-mini-lewm` | a JEPA world model from pixels | encoder, predictor, loss, a report | a target that makes prediction trivial |
| Fri | `d5-planning-surprise` | planning in latent space; surprise | a CEM planner, a surprise score | what the planner refits on |

## The idea of the week

A **world model** predicts what happens next: given the present and an action, the future. With
one, a robot can try actions in imagination before moving (planning), and notice when reality
does not match its prediction (monitoring).

Predicting future **pixels** is wasteful: most pixels are background, lighting, texture, things
that do not matter for control. A **JEPA** (joint-embedding predictive architecture) predicts in
a learned **latent** space instead:

```
image now  ──► encoder ──► z_t ─┐
                                 ├─► predictor ──► predicted z_{t+1}   ← compare →   z_{t+1} ◄── encoder ◄── next image
action     ─────────────────────┘
```

The loss compares a predicted latent with an encoded latent. Nothing is ever decoded back to
pixels. The encoder is free to keep what helps prediction and drop the rest.

That freedom has a cost, and it is the subject of the first two days: the easiest way to make two
latents equal is to make **all** latents the same constant. That is **collapse**.

All code on this page assumes:

```python
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Day 1: Collapse and VICReg

Folder: `d1-collapse-vicreg`. Exercise: `w7d1_vicreg.py`.

### The trap

Take two views of the same thing: the same image with two different noises, or frame `t` and
frame `t + 1`. Ask an encoder `f` to give them nearby latents: `loss = ||f(x1) − f(x2)||²`.

The encoder finds the trivial solution at once: `f(x) = constant`. The loss is exactly zero and
the latent carries no information. The loss curve looks perfect. You will watch it happen today.

Collapse can be **partial**: the latent has 8 dimensions and only one of them varies, or all of
them carry the same number. The loss still looks fine.

### VICReg: three terms

VICReg (variance, invariance, covariance) adds two terms that make the trivial solution
expensive. `z1`, `z2` are the latents of the two views, shape `(N, d)`.

```
invariance = MSE(z1, z2)                                                     the original objective
variance   = mean over dimensions of relu( 1 − sqrt( Var(z_j) + eps ) )      for z1 and for z2
covariance = sum of the squared OFF-diagonal entries of Cov(z), divided by d  for z1 and for z2
loss = 25 · invariance + 25 · variance + 1 · covariance
```

**Variance.** For each latent dimension `j`, take its standard deviation **across the batch**.
If it is below 1, pay the shortfall; at 1 or above, pay nothing (`relu` cuts at zero). A
constant encoder has a standard deviation of 0 in every dimension and pays the maximum. The
variance is over the batch axis, one number per dimension: `z.var(dim=0)`. Over the other axis
it measures something unrelated and does not prevent collapse. That is today's review bug.

**Covariance.** The variance term alone can be satisfied by 8 dimensions that all carry the
same signal. The covariance matrix says how much each pair of dimensions moves together:

```python
z = torch.randn(100, 4)
zc = z - z.mean(dim=0)                         # centre each dimension
cov = zc.T @ zc / (z.shape[0] - 1)             # (d, d); the diagonal holds the variances
off = cov - torch.diag(torch.diag(cov))        # zero the diagonal: what is left is the redundancy
```

Penalising the off-diagonal entries pushes the dimensions to carry **different** information.

### Measuring collapse: effective rank

One number for "how many dimensions does the latent really use".

1. Centre the latent matrix `(N, d)` (subtract the mean of each dimension).
2. Take its **singular values** `s`: one per dimension, each the spread of the data along one
   direction, sorted from largest to smallest. `torch.linalg.svdvals(zc)`.
3. Normalise them into a distribution, `p = s / sum(s)`.
4. `erank = exp( − Σ p · log p )`: the exponential of the entropy of `p`.

If one direction holds everything, `p = (1, 0, 0, …)` and the effective rank is 1. If all `d`
directions are equal, it is `d`. A small `eps` inside the log avoids `log 0`.

A second, cheaper check to print in every JEPA run: the standard deviation of each latent
dimension across the batch. If it heads to 0 while the loss falls, the model is collapsing.

### Traps of the day

- Variance or covariance computed over the dimension axis instead of the batch axis.
- The covariance penalty including the diagonal.
- Not centring before the covariance or the singular values.
- Reading a low loss as success.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `vicreg_loss` | the three terms; `use_cov=False` drops the last | VICReg |
| `effective_rank` | from the singular values of the centred latents | measuring collapse |
| `train_siamese` | one encoder, two noisy views, three kinds of loss | the trap |

Run: `uv run pytest weeks/w7-jepa-world-models/d1-collapse-vicreg -k "not review"`

---

## Day 2: SIGReg, EMA targets, stop-gradient

Folder: `d2-sigreg-ema`. Exercise: `w7d2_sigreg.py`.

There are two families of answers to collapse. You build both today.

### Family 1: regularise the latent distribution (SIGReg)

VICReg constrains variance and covariance. **SIGReg**, the regulariser of LeJEPA and LeWM, asks
for more with a single term: the latents should look like a standard Gaussian `N(0, I)`.

Testing "is this cloud of points in `d` dimensions Gaussian" directly is hard. SIGReg uses a
fact: if `z ~ N(0, I)`, then its projection on **any** unit direction is `N(0, 1)`. So:

1. Draw `M` random unit directions.
2. Project the batch on each: `M` one-dimensional samples.
3. Test each 1-D sample against `N(0, 1)`, and average.

```python
g = torch.Generator().manual_seed(0)
z = torch.randn(256, 8)
U = torch.randn(8, 16, generator=g)
U = U / U.norm(dim=0, keepdim=True)            # (d, M): each COLUMN has length 1
p = z @ U                                      # (N, M): column m is the batch projected on direction m
```

A 1-D test needs few samples, so this works in high dimensions with ordinary batch sizes, and
there is one weight to tune.

**The 1-D test: Epps–Pulley.** It compares the **characteristic function** of the sample with
that of `N(0, 1)`. The characteristic function of a distribution is `φ(t) = E[exp(i·t·x)]`, a
complex number for each `t`; two distributions are equal exactly when their characteristic
functions are. For a sample it is an average, with a real and an imaginary part:

```
re(t) = mean over the sample of cos(t · x)
im(t) = mean over the sample of sin(t · x)
```

For `N(0, 1)`: real part `exp(−t²/2)`, imaginary part 0. The statistic is the squared gap
between the two, weighted and integrated over `t`:

```
t  : K points from −T to T                    (17 points in [−3, 3])
EP = N · ∫ [ (re(t) − e^{−t²/2})² + im(t)² ] · e^{−t²/2} dt
```

The integral is computed with the trapezoid rule: `torch.trapezoid(values, t)` along the `t` axis.

Shapes: `p` is `(N, M)`, `t` is `(K,)`. `t · x` for every sample, direction and `t` is a
broadcast to `(N, M, K)`; the mean over the sample leaves `(M, K)`; the integral over `t` leaves
`(M,)`.

Why it catches collapse: a collapsed latent projects to a constant `c`. Then `re(t) = cos(t·c)`
and `im(t) = sin(t·c)`, far from `exp(−t²/2)` and 0. Large penalty.

### Family 2: an asymmetric target (EMA and stop-gradient)

BYOL, I-JEPA and V-JEPA do not regularise the latent. They break the symmetry between the two
sides of the loss.

- An **online** encoder and a **predictor** are trained by gradient descent.
- A **target** encoder produces what the predictor must match. It receives **no gradient**.
- The target's weights follow the online weights slowly, as an **exponential moving average (EMA)**:

```
θ_target ← m · θ_target + (1 − m) · θ_online          m close to 1: 0.99 to 0.999
```

With `m = 0.99` the target keeps 99% of itself and takes 1% of the online weights at each step.
The two weights swapped give a target that copies the online encoder almost entirely at every
step: no slow target at all. That is today's review bug.

```python
online = nn.Linear(3, 3)
target = copy.deepcopy(online)                       # a separate copy, not a second name
for p in target.parameters():
    p.requires_grad_(False)                          # never trained by the optimiser
with torch.no_grad():                                # an update, not part of any graph
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.mul_(0.5).add_(po, alpha=0.5)             # in place: pt ← 0.5·pt + 0.5·po
```

**Stop-gradient.** The target's output enters the loss as a constant: computed inside
`torch.no_grad()` or passed through `.detach()` (week 1, day 3). The loss can then only be
lowered by moving the prediction toward the target, never by moving the target toward the
prediction. Together with the slow EMA, this avoids collapse in practice, though with no
guarantee, and with more hyper-parameters than SIGReg. LeWM drops both and uses SIGReg alone.

### Traps of the day

- Directions normalised over the wrong axis.
- `m` and `1 − m` swapped.
- The target updated outside `no_grad`, or sharing its parameters with the online encoder.
- A gradient that reaches the target encoder: check `p.grad is None` on its parameters.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `epps_pulley` | `(N, M)` projections → `(M,)` statistics | the 1-D test |
| `sigreg` | random unit directions, project, mean statistic | family 1 |
| `ema_update` | in-place moving average of the target's parameters | family 2 |
| `jepa_ema_loss` | MSE between prediction and target, no gradient into the target | family 2 |

Run: `uv run pytest weeks/w7-jepa-world-models/d2-sigreg-ema -k "not review"`

---

## Day 3: Probes and baselines

Folder: `d3-probes-baselines`. Exercise: `w7d3_probes.py`.

### Why this matters

A world model's loss going down proves nothing: collapse also makes it go down. You need
instruments that look at the latent from outside. Three cheap ones, always used together.

### Instrument 1: the linear probe

Freeze the encoder. Take a quantity `y` you know for each frame (the position of the object, the
gripper opening). Fit the simplest possible map from latent to `y`, a linear one, on **training
episodes**. Measure how well it predicts `y` on **held-out episodes**. If a linear map can read
`y` out of `z`, the latent contains `y` in an easy form.

The fit has a closed form, **ridge regression**. Append a column of ones to the latents so the
bias is learned with the weights:

```
X = [Z, 1]                                (N, d + 1)
[W; b] = (Xᵀ X + λ I)⁻¹ Xᵀ Y              (d + 1, k): the first d rows are W, the last row is b
```

`λ` is a small penalty that keeps the solution stable when latent dimensions are correlated.

```python
A = torch.tensor([[2.0, 0.0], [0.0, 4.0]])
B = torch.tensor([[2.0], [4.0]])
torch.linalg.solve(A, B)                   # the X with A X = B; never invert a matrix yourself
```

### R²: the score of a probe

```
R² = 1 − Σ (y − ŷ)² / Σ (y − ȳ)²
```

The numerator is the probe's squared error. The denominator is the squared error of the
dumbest predictor, the mean `ȳ` of the **true** values on the evaluated set. `R² = 1` is a
perfect fit, `0` is no better than the mean, negative is worse than the mean. Sums run over
samples and dimensions.

The denominator uses the true `y`, not the predictions. With the predictions in the
denominator a constant probe scores a nonsense value. That is today's review bug.

### Instrument 2: the MLP probe

The same test with a small MLP in place of the linear map. Read the two together:

| Linear R² | MLP R² | Meaning |
|---|---|---|
| high | high | the information is there, in a simple form |
| low | high | it is there, encoded nonlinearly (entangled) |
| low | low | it is not there |

A detail of the exercise: the probe is trained inside code that may run under
`torch.no_grad()`. Training needs gradients, so the training loop is wrapped in
`with torch.enable_grad():`, and the latents are passed as `Z.detach()` so that the probe never
sends a gradient into the encoder.

### Instrument 3: the copy-last baseline

For a predictor: how large is the error of predicting "nothing changes", `z_{t+1} = z_t`? In a
slow scene that error is already small. A predictor with a small error has learned something
about dynamics only if it clearly beats copy-last on the same data. Report the **ratio**.

For latent trajectories `(E, T + 1, d)`, the consecutive pairs are two slices along time:
`Z[:, 1:]` and `Z[:, :-1]`.

### Traps of the day

- Probes fitted and scored on the same episodes.
- R² normalised by the predictions, or with `ȳ` taken from the training set.
- No bias column in the ridge.
- A prediction error reported with no copy-last next to it.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `ridge_probe` | closed-form `W`, `b` | instrument 1 |
| `apply_probe` | `Z W + b` | instrument 1 |
| `r2` | the score | R² |
| `mlp_probe` | a small trained MLP, safe under `no_grad` | instrument 2 |
| `copy_last_error` | mean squared change between consecutive latents | instrument 3 |

Run: `uv run pytest weeks/w7-jepa-world-models/d3-probes-baselines -k "not review"`

---

## Day 4: A world model from pixels

Folder: `d4-mini-lewm`. Exercise: `w7d4_lewm.py`. Provided: `pixel_world.py` (read it first),
`jepa_ref.py`, `probes_ref.py`.

### The world

32×32 grey images of a blob. The action is a 2-D displacement and the blob moves by it. An
episode is a sequence of images and actions. The true position of the blob is known, which is
what makes the probes of day 3 possible.

### The model: LeWM's recipe at toy size

```
encoder    image (1, 32, 32) → 3 conv layers (stride 2) → Flatten (512) → Linear → ReLU → Linear → z  (D = 8)
predictor  z_{t+1}_hat = z_t + MLP( concat(z_t, a_t / AMAX) )
loss       MSE( z_{t+1}_hat, encoder(o_{t+1}) )  +  λ · SIGReg( cat(z_t, z_{t+1}) )
```

- **One encoder for both sides, trained end to end.** No EMA, no stop-gradient. The gradient of
  the prediction loss flows into the encoder through `z_t` **and** through the target `z_{t+1}`.
  SIGReg alone prevents collapse.
- **A residual predictor.** It predicts the **change** of the latent. Between two consecutive
  frames the change is small, so "predict zero change" is a good starting point, and it is
  exactly copy-last.
- **The action is scaled** by its maximum, `AMAX`, so the network sees values around [−1, 1].
- **SIGReg on both latents together**, concatenated along the batch axis.

Each conv layer with kernel 4, stride 2, padding 1 halves the resolution: 32 → 16 → 8 → 4.
With 32 channels at the end, `32 · 4 · 4 = 512` numbers go into the first linear layer.

### The experiment, in this order

**1. λ = 0.** Train with the prediction loss alone. Print the standard deviation of `z` and the
prediction loss as training goes. Both go to zero. The model predicts the next latent
perfectly, because every latent is the same.

**2. λ = 0.1.** Train again. Now measure, on **held-out episodes**:

| Number | What it tells you | Instrument |
|---|---|---|
| `z_std` | the latent has not collapsed | the cheap check of day 1 |
| `pred_err`, `copy_err`, their ratio | the predictor learned dynamics | copy-last, day 3 |
| `lin_r2`, `mlp_r2` for the position | the latent knows where the blob is | probes fitted on train, scored on held-out |

A prediction loss is only meaningful next to those numbers. This table is the minimum report
for any world model, yours on the Orin included.

Write the numbers in `PROGRESS.md` and compare them with the numbers of your real models in the
register.

### The planted bug, as a warning

A prediction target that is the **same frame** as the input makes the task trivial: the
residual predictor outputs zero and the loss is tiny. The loss cannot tell you. Only the ratio
to copy-last and a look at the indices can. Always check which frame is the target.

### Traps of the day

- Target and input from the same time step, or from different episodes.
- The action not scaled.
- SIGReg applied when `λ = 0` (wasted compute), or on one side only.
- Probes scored on training episodes.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `Encoder` | image → `z` | the model |
| `Predictor` | residual MLP on `(z, a / AMAX)` | the model |
| `lewm_loss` | prediction MSE + `λ` · SIGReg; returns both | the model |
| `train_lewm` | random `(o_t, a_t, o_{t+1})` triples, Adam on both modules | the model |
| `latent_report` | the table above | the experiment |

Run: `uv run pytest weeks/w7-jepa-world-models/d4-mini-lewm -k "not review" -s` (about 1.5 minutes)

---

## Day 5: Planning and surprise

Folder: `d5-planning-surprise`. Exercise: `w7d5_plan.py`. It imports your day-4 model
automatically. Provided: `pixel_world.py`, `jepa_ref.py`, `probes_ref.py`.

A world model is useful in two roles. You build both.

### Role 1: planner

The model answers "if I do this, what happens?". **Planning** searches for the action sequence
whose predicted outcome is best. No policy is trained.

**Rolling out.** From the current latent `z_0` and a candidate sequence of `H` actions, apply
the predictor `H` times, each time on its own previous output:

```
z_1 = pred(z_0, a_0);  z_2 = pred(z_1, a_1);  …;  z_H = pred(z_{H−1}, a_{H−1})
```

The **cost** of the sequence is how far the predicted latents are from the goal, summed over
the `H` steps.

**CEM, the cross-entropy method.** A simple search that needs no gradient:

```
mu = 0, std = AMAX                                        shape (H, 2): one Gaussian per step of the plan
repeat a few iterations:
    sample N action sequences from N(mu, std), clamp to [−AMAX, AMAX]
    roll each one out, compute its cost
    keep the K sequences with the lowest cost: the elites
    mu, std ← mean and standard deviation of the ELITES (+ a small floor on std)
return mu
```

Each iteration narrows the Gaussian around the best candidates.

```python
N, H, K = 64, 5, 8
mu, std = torch.zeros(H, 2), torch.ones(H, 2)
cand = mu + std * torch.randn(N, H, 2)                 # (N, H, 2)
cost = cand.pow(2).sum(dim=(1, 2))                     # (N,): a stand-in cost
elite = cand[cost.topk(K, largest=False).indices]      # (K, H, 2): the K cheapest
```

Refitting on **all** the samples instead of the elites leaves the Gaussian where it was: the
search never improves. That is today's review bug.

In the exercise everything has one more leading axis, `E`, for the episodes planned in
parallel: candidates are `(E, N, H, 2)`, and the predictor is called on all `E · N` candidates
at once, never in a Python loop over candidates.

**Receding horizon (MPC).** Execute only the **first** action of the plan, observe the new
image, plan again. Errors of the model are corrected at each step by reality.

**Two costs, the same predictor.**

- **Latent L2**: `||z_h − encoder(goal image)||²`. It assumes that distance in latent space
  means distance in the world. Nothing in the training asked for that.
- **Probe-decoded**: decode the predicted latent to a position with an MLP probe (day 3), and
  measure the distance to the goal position.

If the second works much better than the first, the dynamics are fine and the problem is the
**metric** of the latent space. That distinction is a central question about JEPA planners.

### Role 2: evaluator (surprise)

The model also says what it expected. The **surprise** of a transition is its prediction error:

```
s_t = || pred(z_t, a_t) − z_{t+1} ||²
```

An unexpected event should make it spike. Today's event is a teleport: in the middle of an
episode the blob jumps, as if someone moved the object. You score every transition and measure,
with the AUROC of week 6, how well the surprise separates the teleport transitions from the
normal ones.

That is a failure monitor built from a world model. On real data the same warning applies as in
week 6: it must beat the cheapest baseline on the same episodes.

### Traps of the day

- Refit on all samples.
- Costs taken on the last latent only when the docstring asks for the sum over the horizon.
- A standard deviation that reaches zero: keep the floor.
- The whole plan executed open loop instead of the first action.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `cem_plan` | batched CEM over action sequences; returns the first action | role 1 |
| `latent_cost` | squared distance to the goal latent | two costs |
| `probe_cost` | squared distance of the decoded position to the goal | two costs |
| `run_planner` | closed loop in the pixel world | receding horizon |
| `surprise_scores` | scores and labels for teleport detection | role 2 |

Run: `uv run pytest weeks/w7-jepa-world-models/d5-planning-surprise -k "not review" -s` (about 1.5 minutes)

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w7` (DP, interval scheduling), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 7 (your real LeWM training script against your mini one).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 7 when you can do these without notes:

- [ ] explain collapse and show the loss that causes it
- [ ] give the three terms of VICReg and what each prevents
- [ ] explain SIGReg in three sentences; explain EMA and stop-gradient in three more
- [ ] say what a linear probe, an MLP probe and copy-last each tell you, and write R²
- [ ] draw a JEPA world model and list the numbers you report with it
- [ ] write CEM from memory and explain receding horizon
- [ ] explain why latent distance may be a bad planning cost

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/05-jepa-ssl.md`, `knowledge/06-world-models-planning.md`
- The papers behind this week, in depth: `reference/journal/book/12-jepa-world-models.md`, `12a-jepa-papers-in-depth.md`, `12b-jepa-planning-and-grounding-papers.md`
- Previous week: [`../w6-robot-learning-eval/README.md`](../w6-robot-learning-eval/README.md) · Next: [`../w8-robotics-register/README.md`](../w8-robotics-register/README.md)
