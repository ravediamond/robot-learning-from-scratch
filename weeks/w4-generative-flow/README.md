# Week 4: Generative models for actions (VAE, diffusion, flow matching)

**Everything you need for this week is on this page.** You need weeks 1–2 (training loop, masked
losses, behavior cloning). Week 3 is not required. Nothing from other folders is required.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [The idea of the week](#the-idea-of-the-week).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-why-mse-fails-and-the-vae),
   [Day 2](#day-2-diffusion-ddpm), [Day 3](#day-3-flow-matching),
   [Day 4](#day-4-conditioning-and-classifier-free-guidance), [Day 5](#day-5-a-flow-policy-against-an-mse-policy-closed-loop).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w4dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w4dN_*.py` | tests | run, never edit |
| `review_w4dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided helpers: `toy2d.py` (2-D datasets, `time_embedding`, a distance metric), `sim2d.py`
(day 5: the obstacle world), `plot_flow.py` (day 3, optional picture).

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w4-generative-flow/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w4-generative-flow/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w4-generative-flow/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w4-generative-flow/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-mse-fails-vae` | why MSE averages the modes; the VAE | a VAE that covers both modes of a bimodal dataset | variance and standard deviation confused |
| Tue | `d2-ddpm` | diffusion: add noise, learn to remove it | a DDPM on two moons | two noise coefficients swapped |
| Wed | `d3-flow-matching-2d` | flow matching: a straight path from noise to data | loss and Euler sampler; the 1-step experiment | the time convention flipped |
| Thu | `d4-conditional-cfg` | conditioning, classifier-free guidance | a class-conditional flow with guidance | guidance weights swapped |
| Fri | `d5-flow-policy-obstacle` | action heads in closed loop | a flow policy and an MSE policy, a rollout loop, rates with intervals | actions not converted back to real units |

## The idea of the week

Until now a policy was a function: one observation in, one action out, trained with MSE. That
breaks as soon as the demonstrations contain **several good answers** for the same observation.

An obstacle in front of the robot; half of the operators go around it on the left, half on the
right. The MSE-optimal prediction is the **average**: straight ahead, into the obstacle. More
data and a bigger network do not help. It is what the loss asks for.

The fix is to stop predicting one action and start modelling the **distribution** of actions,
`p(action | observation)`, and to **sample** from it. A sample is one of the good answers, never
their average. A model you can sample from is a **generative model**. This week you build three:
the VAE (inside ACT), diffusion (inside Diffusion Policy) and flow matching (inside π0 and SmolVLA).

All three share one trick: they turn **random noise** into data. Noise is easy to sample; the
network learns the transformation.

All code on this page assumes:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Day 1: Why MSE fails, and the VAE

Folder: `d1-mse-fails-vae`. Exercise: `w4d1_vae.py`.

### MSE predicts the mean

Take targets `y₁ … y_N` and one constant prediction `c`. The loss is `mean((yᵢ − c)²)`. Its
derivative with respect to `c` is `−2 · mean(yᵢ − c)`, which is zero when `c = mean(y)`. So the
best constant under MSE is the mean. A network trained with MSE does the same thing for each
observation: it outputs the mean of the actions seen for that observation.

Targets −1 and +1, half each: the optimum is 0, a value no operator ever chose.

### The VAE

A **variational auto-encoder** has two networks and a small random vector `z`, the **latent**.

- The **decoder** turns a latent `z` into a data point. To generate: draw `z ~ N(0, I)`, decode.
  Different `z` give different outputs, so the model can produce both modes.
- The **encoder** is only used in training. It looks at a real data point `x` and proposes which
  `z` would decode into it, as a small Gaussian: a mean `mu` and a log-variance `logvar`.

```
encoder(x) → mu, logvar                        q(z | x) = N(mu, exp(logvar))
z = mu + exp(0.5 · logvar) · eps               eps ~ N(0, I)
decoder(z) → x_hat
loss = reconstruction(x_hat, x) + beta · KL( q(z | x) || N(0, I) )
```

**Why `logvar`.** A variance must be positive, and a network outputs any real number. So the
network outputs the **log** of the variance. Variance is `exp(logvar)`; the **standard
deviation** is its square root, `exp(0.5 · logvar)`. Mixing the two is today's review bug.

**The reparameterisation trick.** Sampling `z` directly is a random draw, and gradients cannot
pass through a random draw. Write the same sample as `mu + std · eps`: the randomness is now in
`eps`, an input, and `z` is an ordinary function of `mu` and `std`. Gradients flow to both.

```python
mu = torch.zeros(4, 2, requires_grad=True)
std = torch.ones(4, 2)
z = mu + std * torch.randn_like(mu)        # a sample, and still differentiable in mu
```

**The two terms of the loss pull against each other.**

- **Reconstruction** wants each `x` to get its own precise `z`, far from the others.
- **KL** wants every `q(z | x)` to look like `N(0, I)`, the distribution you sample from at
  generation time. Without it, the latents used in training would live somewhere the generator
  never samples.

For two Gaussians the KL has a closed form, per sample, summed over the latent dimensions:

```
KL = 0.5 · Σ ( exp(logvar) + mu² − 1 − logvar )
```

It is 0 when `mu = 0` and `logvar = 0`, and positive otherwise. `beta` sets the balance: too
high, and the encoder gives up and outputs `N(0, I)` for everything (the decoder then ignores
`z` and outputs the mean again: **posterior collapse**).

**Reductions matter.** The reconstruction term and the KL must be on the same scale: both
summed over their dimensions per sample, then averaged over the batch. A `mean` over dimensions
on one term silently changes the effective `beta`.

**ACT is a conditional VAE.** In training its encoder sees the observation and the true action
chunk; its decoder sees the observation and `z`. At test time `z = 0`. You build it in week 6.

### Traps of the day

- `exp(logvar)` used where the standard deviation is needed.
- `eps` drawn once and reused: draw it fresh at every step.
- Reconstruction averaged, KL summed.
- Sampling at generation time through the encoder. Generation uses `z ~ N(0, I)` and the decoder only.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `mse_optimum` | the best constant prediction, in closed form | MSE predicts the mean |
| `reparameterize` | `mu + std · eps` | the reparameterisation trick |
| `kl_to_standard_normal` | closed-form KL per sample | the two terms of the loss |
| `VAE` | encoder → `mu`, `logvar`; decoder | the VAE |
| `vae_loss` | reconstruction + `beta` · KL | the two terms, reductions |
| `train_vae` | Adam, fresh noise every step, returns in eval mode | weeks 1–2 |

Run: `uv run pytest weeks/w4-generative-flow/d1-mse-fails-vae -k "not review"`

---

## Day 2: Diffusion (DDPM)

Folder: `d2-ddpm`. Exercise: `w4d2_ddpm.py`. Provided: `toy2d.py`.

### The idea

Destroying data is easy: add a little Gaussian noise, again and again, until nothing is left but
noise. **Diffusion** trains a network to undo one small step of that destruction. To generate,
start from pure noise and undo the steps one by one.

Diffusion Policy is exactly this, with action chunks in place of today's 2-D points.

### The forward process: adding noise

`T` steps (say 100 to 1000). Step `t` mixes in a small amount of noise, set by `beta_t`:

```
betas   : T values rising linearly from 1e-4 to 0.02          how much noise each step adds
alpha_t = 1 − beta_t                                           how much signal each step keeps
abar_t  = alpha_0 · alpha_1 · … · alpha_t                     signal kept after t+1 steps (cumulative product)
```

Because every step is Gaussian, `t` steps collapse into one formula. You can jump from clean
data `x_0` to the noisy `x_t` directly:

```
x_t = sqrt(abar_t) · x_0  +  sqrt(1 − abar_t) · eps          eps ~ N(0, I)
```

`abar` goes from almost 1 (nearly clean) down to almost 0 (nearly pure noise). The two
coefficients are square roots because **variances** add: `abar + (1 − abar) = 1`, so `x_t`
keeps a variance of about 1 at every step.

```python
betas = torch.linspace(1e-4, 0.02, 5)
torch.cumprod(1 - betas, dim=0)            # the running product: decreasing from ~1
```

Indices are 0-based in this lab: `abar[0]` is after **one** noising step.

**One coefficient per sample.** In a batch, each sample has its own `t`, a long tensor `(N,)`.
`abar[t]` is `(N,)` and must multiply `x_0` of shape `(N, D)`: give it a last axis, `abar[t][:, None]`.

### Training: predict the noise

Draw a data point, a random step `t` and a noise `eps`. Build `x_t` with the formula. Ask the
network to recover the noise that was added:

```
loss = mean( ( eps − eps_theta(x_t, t) )² )
```

Plain MSE again, and no averaging-of-modes problem: for a given `x_t` and `t`, the noise is a
well-defined regression target.

**The network must know `t`.** Removing noise from an almost clean point and from an almost
random one are different jobs. `t` is a number; feeding it raw works badly. It is expanded into
a vector of sines and cosines at several frequencies, as positions were in week 3:
`toy2d.time_embedding(t / T, 32)`. That vector is concatenated with `x`.

### Sampling: remove the noise step by step

```
x ~ N(0, I)
for t = T−1 down to 0:
    eps_hat = eps_theta(x, t)
    x = ( x − beta_t / sqrt(1 − abar_t) · eps_hat ) / sqrt(alpha_t)        remove the predicted noise
    if t > 0:  x = x + sqrt(beta_t) · z                 z ~ N(0, I): put a little fresh noise back
```

Each step removes the predicted noise and rescales. Fresh noise is added back at every step
except the last: the sampler is random at each step, which is called **ancestral sampling**.
Sampling runs inside `torch.no_grad()` and the same `t` is given to the whole batch:
`torch.full((N,), t, dtype=torch.long)`.

The cost is the weak point: one network call per step, hundreds of steps. That is why
flow matching (tomorrow) replaced it in recent robot policies.

### Traps of the day

- `sqrt(abar)` and `sqrt(1 − abar)` swapped: the code runs and learns nothing useful.
- `abar[t]` not broadcast over the data dimension.
- Noise added at the last sampling step.
- `t` given to the network as an integer in 0..T−1 when the embedding expects `t / T`.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `linear_betas`, `alphas_bar` | the schedule | the forward process |
| `q_sample` | `x_0`, `t`, noise → `x_t` | the forward process |
| `Denoiser` | MLP on `concat(x, time_embedding)` | training |
| `ddpm_loss` | random `t`, random noise, MSE on the noise | training |
| `sample` | ancestral sampling from noise | sampling |
| `train_ddpm` | the loop; returns the model and the betas | weeks 1–2 |

Run: `uv run pytest weeks/w4-generative-flow/d2-ddpm -k "not review"`

---

## Day 3: Flow matching

Folder: `d3-flow-matching-2d`. Exercise: `w4d3_flow.py`. Provided: `toy2d.py`, `plot_flow.py`.

### The idea

Same goal as diffusion, with a simpler path. Take a noise point `x0` and a data point `x1`, and
draw the **straight line** between them. A point moving along that line at constant speed has
the velocity `x1 − x0`. Train a network to predict that velocity. To generate, start from noise
and follow the predicted velocity.

This is the loss and the sampler inside the action expert of π0 and SmolVLA.

### The convention of this course: t = 0 is noise, t = 1 is data

```
x0 ~ N(0, I)           noise
x1                     a data point
t  ~ U(0, 1)           one value per sample
x_t = (1 − t) · x0 + t · x1          a point on the line
target velocity = x1 − x0            the same at every t
loss = mean( ( v_theta(x_t, t) − (x1 − x0) )² )
```

Papers and codebases disagree on which end is noise. Before you read or write any flow code,
check: at `t = 0`, is `x_t` the noise? Getting it backwards is today's review bug, and it does
not crash.

`t` has shape `(N,)` and multiplies `(N, D)`: `t[:, None]`, as yesterday.

### Why it works when one line crosses another

Many (noise, data) lines pass through the same point `x_t` and they disagree on the velocity.
MSE gives the network their **average** at that point. Following that average velocity field
from a noise sample still ends on the data: near `t = 0` all lines agree roughly (go toward the
data), and as the point moves, fewer and fewer lines pass through where it is, until only the
lines of one mode remain.

### Sampling: Euler integration

The network defines a velocity at every point and time. Generating is solving
`dx/dt = v(x, t)` from `t = 0` to `t = 1`. The simplest solver takes `N` equal steps:

```
x = x0                                  noise
for i in 0 … N−1:
    t = i / N
    x = x + (1 / N) · v_theta(x, t)
```

"Where I am, plus a small step in the direction of the velocity." Ten steps are typical for a
robot policy, against hundreds for DDPM, and there is no noise added along the way: the same
`x0` always gives the same output. Sampling runs inside `torch.no_grad()`.

### The one-step experiment

With **one** Euler step from `t = 0`, the network answers "where does this noise go?" before
the point has moved at all. At `t = 0`, `x_t` is pure noise and says nothing about which data
point it is paired with. The best MSE answer is `mean(x1) − x0`, and one step lands on
`x0 + mean(x1) − x0 = mean(x1)`: **the mean of the data**. One step is MSE regression again.

More steps let the trajectory commit to a mode. You measure this today; it is the clearest way to
see why the number of sampling steps matters, and what a "one-step policy" has to solve.

### Traps of the day

- The convention flipped in the interpolation, the target, or the sampler. All three must agree.
- One `t` for the whole batch in training. Draw one per sample.
- `t` not broadcast.
- The loop evaluating the velocity at `t = (i + 1) / N` instead of the start of the step.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `fm_interpolate` | `x0`, `x1`, `t` → `x_t` and the target velocity | the convention |
| `VelocityNet` | MLP on `concat(x, time_embedding(t))` | day 2, the network must know `t` |
| `fm_loss` | fresh noise and `t` per sample, MSE on the velocity | the convention |
| `euler_sample` | integrate from 0 to 1 in `steps` steps | sampling |
| `train_fm` | the loop | weeks 1–2 |

Run: `uv run pytest weeks/w4-generative-flow/d3-flow-matching-2d -k "not review"`
Optional: `uv run python weeks/w4-generative-flow/d3-flow-matching-2d/plot_flow.py` saves a picture of data, 1-step and 10-step samples.

---

## Day 4: Conditioning and classifier-free guidance

Folder: `d4-conditional-cfg`. Exercise: `w4d4_cfg.py`. Provided: `toy2d.py`.

### Conditioning

Yesterday's model samples "a data point". A policy must sample "an action **for this
observation**", and a VLA "an action for this image and this instruction". The change is small:
the network gets one more input, the condition `c`.

```
v(x, t, c)
```

Today the condition is a class label, an integer. It enters through an `nn.Embedding`, a lookup
table from label to vector, concatenated with `x` and the time embedding. The loss is
yesterday's, computed on (data, label) pairs.

### Classifier-free guidance (CFG)

A conditional model sometimes follows its condition only weakly. **Guidance** makes it follow
the condition more, at sampling time, with no extra network.

**Training.** Reserve one extra label, the **null** label, meaning "no condition". For each
training sample, with a small probability `p` (about 0.1), replace its label by null. One
network then learns both the conditional velocity `v(x, t, c)` and the unconditional one
`v(x, t, ∅)`.

```python
labels = torch.tensor([0, 1, 2, 3])
drop = torch.rand(4) < 0.1                       # one independent coin per sample
torch.where(drop, torch.tensor(4), labels)       # a NEW tensor; 4 plays the null label here
```

With `n_classes` real labels, the embedding table has `n_classes + 1` rows, and the last index
is null.

**Sampling.** At each step, call the network twice, with the label and with null, and
extrapolate from the unconditional velocity through the conditional one:

```
v = v_uncond + w · ( v_cond − v_uncond )
```

| `w` | Result |
|---|---|
| 0 | unconditional: the label is ignored |
| 1 | the plain conditional model |
| > 1 | pushed further toward what the condition adds |

`v_cond − v_uncond` is "what the condition changes". `w > 1` exaggerates it: samples match the
condition better and vary less. The cost is two network calls per step.

Which of the two velocities gets the weight `w` is easy to swap; check `w = 0` and `w = 1`
by hand in your code.

### Traps of the day

- An embedding table with `n_classes` rows: the null index is out of range.
- Label dropout applied in place on the caller's tensor, or at sampling time.
- `v_cond` and `v_uncond` swapped in the guidance formula.
- Dropping the whole batch's labels with one coin.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `CondVelocityNet` | `concat(x, time_embedding, embed(c))` → MLP; `self.null` | conditioning |
| `drop_labels` | independent replacement by null, new tensor | CFG, training |
| `cond_fm_loss` | flow-matching loss with label dropout | CFG, training; day 3 |
| `cfg_velocity` | the guidance formula | CFG, sampling |
| `cfg_sample` | Euler with the guided velocity | CFG, sampling; day 3 |
| `train_cond_fm` | the loop | weeks 1–2 |

Run: `uv run pytest weeks/w4-generative-flow/d4-conditional-cfg -k "not review"`

---

## Day 5: A flow policy against an MSE policy, closed loop

Folder: `d5-flow-policy-obstacle`. Exercise: `w4d5_flow_policy.py`. Provided: `sim2d.py` (read
it first, 80 lines), `toy2d.py`.

### The world

A point robot in a square. It starts on the left; the goal is on the right; a round obstacle
sits in the middle. The demonstrations go **around** it, half above and half below: bimodal, as
in the idea of the week.

- Observation: position `(2,)`.
- Action: a velocity `(2,)`; `pos ← pos + action`, with the action clipped to `MAX_STEP`.
- Action chunk: the next `H = 8` velocities.

### Two heads, same everything else

| | Input | Output | Trained with |
|---|---|---|---|
| **MSE head** | observation | the chunk `(N, H, 2)` | masked MSE (week 2, day 5) |
| **Flow head** | observation + noisy chunk `x_t` + `t` | a velocity `(N, H, 2)` | flow matching (day 3), conditioned on the observation (day 4) |

The flow head is day 3 with two changes: the "data point" is a whole chunk, flattened to `H·2`
numbers for the MLP, and the observation is concatenated as the condition. To act: start from
noise of shape `(N, H, 2)`, take 10 Euler steps, get a chunk.

**Units.** Train both heads on actions divided by `MAX_STEP`, so the values are about in
[−1, 1], the scale of the noise. To act, multiply back. Forgetting the conversion gives a robot
that barely moves or flies off: today's review bug.

### Open loop and closed loop

A validation loss compares predictions with recorded actions: **open loop**. It cannot see the
problem of the week. The MSE head has a perfectly reasonable validation loss while it drives
into the obstacle.

**Closed loop** means running the policy in the world: its own actions decide what it observes
next. That is the only measurement that counts, and you write the loop yourself.

### The rollout

**Receding horizon.** The policy predicts `H = 8` actions; you execute the first `H_exec = 4`,
then ask for a new chunk from where the robot now is. Replanning often corrects errors;
executing several actions per call keeps the motion consistent and saves compute.

```
positions ← starts (n, 2);  every episode active
repeat until T_MAX steps:
    every H_exec steps: new chunks for the episodes that are still active
    take the next action of each chunk, clip it, move
    mark collisions, then successes (reached the goal and never collided)
    an episode that collided or succeeded stops
```

It is batched over episodes: all `n` episodes advance together as tensors, with a boolean
`active` mask. Finished episodes must stop moving and must not change their outcome.
`sim2d` gives you `sample_starts`, `clip_action`, `collides`, `reached`.

### Reporting a rate honestly

"172 successes out of 200" is an estimate of the true success rate, with uncertainty. The
**Wilson interval** gives a 95% range for a rate measured as `k` successes in `n` trials:

```
z = 1.96
centre     = (k + z²/2) / (n + z²)
half-width = z · sqrt( k·(n − k)/n + z²/4 ) / (n + z²)
```

7 successes out of 10 gives (0.40, 0.89): very wide. That is why the test uses 200 episodes
per policy. `sim2d.wilson(k, n)` is provided; week 6 has you write it.

Write your prediction first: which head collides more, and by how much? Then run. The test bars:
MSE collision rate above 30%; flow collision below 15% and success above 85%.

### Traps of the day

- Actions not multiplied back by `MAX_STEP`, or not clipped.
- Finished episodes that keep moving.
- Chunks requested for every episode at every step.
- A different random generator for the starts and for the policy than the docstring asks.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `MSEChunkPolicy` | obs → chunk; `.act` returns raw actions | two heads |
| `FlowChunkPolicy` | `v(x_t, t, obs)`; `.act` samples with Euler steps | two heads; days 3–4 |
| `masked_mse` | mean over real steps | week 2, day 5 |
| `train_mse_policy`, `train_flow_policy` | the two loops, in units of `MAX_STEP` | units |
| `rollout` | batched closed-loop episodes with replanning | the rollout |

Run: `uv run pytest weeks/w4-generative-flow/d5-flow-policy-obstacle -k "not review" -s` (about 1 minute)

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w4` (BFS and DFS on grids), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 4 (SmolVLA's flow loss and sampler: which end is noise there?).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 4 when you can do these without notes:

- [ ] show in two lines why MSE predicts the mean, and what that does to bimodal demonstrations
- [ ] write the VAE loss and explain reparameterisation and the role of the KL
- [ ] write the DDPM forward formula and the training loss
- [ ] write the flow-matching loss and the Euler sampler, with the convention stated
- [ ] explain why one Euler step gives the mean
- [ ] write the guidance formula and say what `w = 0`, `1`, `2` do
- [ ] write a batched closed-loop rollout and report a rate with its interval

## Optional depth (not needed for the labs)

- Interview questions on diffusion and flow matching: `knowledge/04-diffusion-flow-matching.md`
- Action heads in real policies: `reference/journal/book/11-policies-chunks-and-boundaries.md` (11.3)
- Timed version for later: `drills/live/L06_flow_matching.py`
- Previous week: [`../w3-transformers-gpt2/README.md`](../w3-transformers-gpt2/README.md) · Next: [`../w5-vlm-vla/README.md`](../w5-vlm-vla/README.md)
