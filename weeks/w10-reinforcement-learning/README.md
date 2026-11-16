# Week 10: Reinforcement learning, from returns to RL on top of imitation

**Everything you need for this week is on this page.** You need week 1 (autograd, training loop,
`detach`), week 4 (reparameterisation, classifier-free guidance) and week 7 day 2 (slow target
networks). Nothing from other folders is required.

Almost every robot-learning job posting names RL next to imitation learning, and the frontier
result of 2025–2026 (π*0.6 with RECAP) is "imitation first, then improve with RL from the robot's
own experience". This week you write the pieces by hand, small enough to run on a Mac in seconds.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [Reinforcement learning in ten minutes](#reinforcement-learning-in-ten-minutes).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-returns-advantages-the-policy-gradient),
   [Day 2](#day-2-ppo), [Day 3](#day-3-sac), [Day 4](#day-4-offline-rl-with-iql),
   [Day 5](#day-5-recap-style-advantage-conditioning).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w10dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w10dN_*.py` | tests | run, never edit |
| `review_w10dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided helpers: `rl_envs.py` (the world, in every folder), `w10d2_helpers.py` (day 1's
functions, so day 2 stands alone), `offline_data.py` (days 4 and 5).

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w10-reinforcement-learning/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w10-reinforcement-learning/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w10-reinforcement-learning/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w10-reinforcement-learning/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-returns-policy-gradient` | returns, advantages, GAE, the policy gradient | REINFORCE with a baseline | returns that leak across an episode boundary |
| Tue | `d2-ppo` | reusing a batch safely | the clipped objective and the update loop | one word in the clip |
| Wed | `d3-sac` | off-policy learning with a replay buffer | the five pieces of SAC | a missing correction in a log-probability |
| Thu | `d4-offline-rl-iql` | learning from a fixed, mixed dataset | IQL, against behavior cloning | two weights swapped |
| Fri | `d5-recap-advantage-conditioning` | RL on top of imitation, as in π*0.6 | advantage-conditioned imitation | one threshold where several are needed |

## Reinforcement learning in ten minutes

**Imitation** learns from what an expert did. **Reinforcement learning (RL)** learns from a
number, the **reward**, that says how good each step was. No expert is needed, and the policy can
become better than its data. The price is that the learner must find out by itself which actions
lead to reward.

| Word | Meaning |
|---|---|
| state `s`, observation | what the agent sees |
| action `a` | what it does |
| reward `r` | one number per step; the only signal |
| policy `π(a \| s)` | the agent: a distribution over actions given the state |
| episode | one run from a start to an end; `done` marks the last step |
| return `G` | the sum of future rewards, discounted |
| discount `γ` | between 0 and 1: a reward `k` steps away counts `γᵏ` |
| value `V(s)` | the expected return from state `s`, following the policy |
| `Q(s, a)` | the expected return after taking action `a` in `s`, then following the policy |
| advantage `A(s, a)` | `Q(s, a) − V(s)`: was this action better or worse than usual here? |
| on-policy | learns only from data collected by the current policy (days 1–2) |
| off-policy | learns from any stored data (day 3) |
| offline | learns from a fixed dataset, with no new interaction (days 4–5) |

**The world of the week**, `rl_envs.py`: `Reach2D`, a batch of points in the plane that must
reach a goal. Observation `(pos, goal)`, 4 numbers. Action: a velocity in `[−1, 1]²`. Reward:
minus the distance to the goal. Fixed horizon of 20 steps. The best return is close to 0.

Tensors of collected experience are **time-major**: shape `(T, N)` means `T` steps of `N`
parallel environments. Row `t` is one moment in time, across all environments.

All code on this page assumes:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Day 1: Returns, advantages, the policy gradient

Folder: `d1-returns-policy-gradient`. Exercise: `w10d1_pg.py`.

### The discounted return

The return from step `t` is `r_t + γ r_{t+1} + γ² r_{t+2} + …` until the episode ends. It
satisfies a recursion, which is how it is computed, **backwards** from the last step:

```
G_t = r_t + γ · (1 − done_t) · G_{t+1}          G after the last stored step = 0
```

`done_t` is True when the episode ended **at** step `t`. The factor `(1 − done_t)` cuts the sum
there: what comes after belongs to another episode. Without it, the rewards of the next episode
leak into this one. That is today's review bug.

```python
r = torch.tensor([1.0, 1.0, 1.0]); gamma = 0.5
G = torch.zeros(3); nxt = 0.0
for t in reversed(range(3)):           # backwards
    nxt = r[t] + gamma * nxt
    G[t] = nxt
G                                      # tensor([1.7500, 1.5000, 1.0000])
```

### The policy gradient (REINFORCE)

A stochastic policy outputs a distribution. Here a **Gaussian**: a network gives the mean, and
one learned `log_std` per action dimension gives the spread. The action is a sample.

The **log-probability** of an action under a diagonal Gaussian, summed over action dimensions:

```
log N(a; μ, σ) = Σ_d [ −½ ((a_d − μ_d) / σ_d)² − log σ_d − ½ log(2π) ]
```

```python
dist = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
dist.log_prob(torch.tensor([0.5, -0.5])).sum()      # your formula must give the same number
```

The idea of the policy gradient: **make the actions that were followed by a high return more
likely**. As a loss:

```
loss = − mean( log π(a_t | s_t) · weight_t )          weight_t = G_t − baseline_t, with NO gradient
```

The gradient flows only through `log π`. The weight is a number that says how much to push; it
must be detached, or the loss would also try to change the weight itself.

### The baseline and the advantage

With raw returns as weights, every action in a good episode is pushed up, good or not, and the
gradient is very noisy. Subtract a **baseline** that does not depend on the action: the gradient
keeps the same expected value and its variance drops. The natural baseline is "the usual return
from here". With `b = V(s)` the weight is the **advantage**: positive for better than usual,
negative for worse.

Today's baseline is the simplest one: the mean return over the batch at each time step, shape
`(T, 1)`, which broadcasts over the `N` environments.

### GAE

With a learned value function there are two ways to estimate an advantage:

- **One step**: `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`. Low noise; wrong when `V` is wrong (bias).
- **Full return**: `G_t − V(s_t)`. No bias from `V`; very noisy.

**GAE** (generalised advantage estimation) blends them with a dial `λ`:

```
δ_t = r_t + γ (1 − done_t) V_{t+1} − V_t
A_t = δ_t + γ λ (1 − done_t) A_{t+1}             computed backwards, like the return
```

`λ = 0` gives the one-step estimate, `λ = 1` the full return minus `V`. `V_{t+1}` for the last
stored step is `last_value`, the value of the state that comes after the stored data (the
**bootstrap**). The same `(1 − done_t)` appears twice: once for the value of the next state, once
for the recursion.

### Traps of the day

- No `(1 − done)`: returns leak across episodes.
- The weight not detached.
- A loop forwards in time.
- The log-probability not summed over the action dimensions: shape `(T, N, A)` instead of `(T, N)`.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `discounted_returns` | `(T, N)` rewards and dones → returns | the discounted return |
| `gae` | advantages from rewards, values, dones, `last_value` | GAE |
| `GaussianPolicy` | mean from an MLP, learned `log_std`; `log_prob` | the policy gradient |
| `reinforce_loss` | `−mean(logp · (returns − baseline))`, weight without gradient | the policy gradient, the baseline |
| `train_reinforce` | collect, returns, baseline, one Adam step, repeat | all of the above |

Run: `uv run pytest weeks/w10-reinforcement-learning/d1-returns-policy-gradient -k "not review"`

---

## Day 2: PPO

Folder: `d2-ppo`. Exercise: `w10d2_ppo.py`. Provided: `w10d2_helpers.py`.

### The idea

REINFORCE uses each batch for one gradient step, then throws it away. Collecting data is the
expensive part, so you want several steps per batch. But after the first step the policy has
changed, and the data no longer comes from it. Take too many steps and the policy moves
somewhere the data says nothing about, and collapses.

**PPO** (proximal policy optimisation) reuses the batch for several epochs and **stops the policy
from moving too far** from the one that collected the data.

### The ratio and the clipped objective

How far the policy has moved, for one action: the ratio of its new probability to its old one.

```
ρ = π_new(a | s) / π_old(a | s) = exp( logp_new − logp_old )
```

`ρ = 1` at the start of the update. The objective to maximise is `ρ · A`: raise the probability
of actions with a positive advantage. PPO's loss:

```
L_clip = − mean( min( ρ · A ,  clamp(ρ, 1 − ε, 1 + ε) · A ) )          ε ≈ 0.2
```

Read the `min` case by case:

| Advantage | `ρ` moved… | What the `min` does |
|---|---|---|
| positive | above `1 + ε` (the helpful way) | picks the clipped term: gradient zero, **stop** |
| negative | below `1 − ε` (the helpful way) | picks the clipped term: gradient zero, **stop** |
| either | the harmful way | picks the unclipped term: full gradient to come back |

The `min` makes the objective **pessimistic**: it never rewards a move beyond the clip, and
never forgives a move in the wrong direction. With `max` in its place the opposite happens.
That is today's review bug.

### The bookkeeping that decides whether it works

- **`logp_old` and the old values are stored at collection time, without gradient.** They are
  constants during the update.
- **Advantages from GAE, normalised per minibatch** to zero mean and unit standard deviation,
  and detached. Normalising keeps the step size stable when the scale of the rewards changes.
- **A value network** trained alongside: `0.5 · (V(s) − target)²`, with
  `target = advantage + old value`.
- **Several epochs over shuffled minibatches.** Flatten `(T, N, …)` to `(T·N, …)`, shuffle the
  indices with `torch.randperm`, cut them into chunks.

```python
g = torch.Generator().manual_seed(0)
perm = torch.randperm(10, generator=g)
perm.chunk(3)                          # three index tensors: the minibatches of one epoch
```

### Traps of the day

- `max` for `min`.
- `logp_old` recomputed with the current policy: the ratio is always 1.
- Advantages with a gradient, or normalised over the whole batch when the docstring says per minibatch.
- One optimiser step per batch: that is REINFORCE again.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `ppo_clip_loss` | the clipped objective | the ratio and the clipped objective |
| `normalize` | zero mean, unit std | the bookkeeping |
| `value_loss` | half the mean squared error | the bookkeeping |
| `ppo_update` | GAE, flatten, epochs, minibatches, one loss, one step each | the bookkeeping |
| `train_ppo` | collect and update, repeated | all of the above |

Run: `uv run pytest weeks/w10-reinforcement-learning/d2-ppo -k "not review"`

---

## Day 3: SAC

Folder: `d3-sac`. Exercise: `w10d3_sac.py`.

### Why off-policy

PPO throws each batch away after a few epochs. On a real arm every transition costs time and
wear. **Off-policy** methods keep every transition `(s, a, r, s', done)` in a **replay buffer**
and learn from all of them, many times. SAC (soft actor-critic) is the standard recipe for
continuous actions, and the one used for RL on real robots (HIL-SERL).

### The critic: learning Q

`Q(s, a)` is learned from one-step consistency. If the estimates were right, then

```
Q(s, a) = r + γ · (1 − done) · Q(s', a')          a' = what the policy would do in s'
```

So the right-hand side, computed with the stored transition, is a regression **target** for the
left-hand side. This is **temporal-difference learning**. It needs no full episode: any stored
transition will do, whoever collected it.

Two stabilisers, both about the target:

- **Target networks.** The target is computed with a slow copy of the critic, updated as a
  moving average (the EMA of week 7): `θ' ← (1 − τ) θ' + τ θ`, with a small `τ`. Otherwise the
  network chases a target that moves with it.
- **Twin critics.** A learned Q is noisy, and a policy that seeks high Q finds the places where
  it is too high. Train two critics and use the **minimum** of the two in the target.

### The actor: a squashed Gaussian

Actions are bounded. The policy samples `u ~ N(μ, σ)` and acts with `a = tanh(u)`, which lies in
`(−1, 1)`. The sample is written `u = μ + σ · ε` (the reparameterisation of week 4), so gradients
flow from the action back to `μ` and `σ`.

Squashing changes the density. The log-probability of `a` is the Gaussian log-probability of `u`
with a correction, summed over action dimensions:

```
log π(a) = log N(u; μ, σ) − Σ log( 1 − tanh²(u) )
```

Without the correction the entropy below is computed for the wrong distribution, and the
temperature drives the policy to the edges of the action range. That is today's review bug.

### Entropy and the temperature

SAC maximises reward **plus entropy**: it prefers policies that stay random where randomness
costs nothing. That keeps exploration alive. `−log π(a | s)` is the entropy of one sample, and
`α`, the **temperature**, is its price.

```
target       y   = r + γ (1 − done) · ( min(Q1', Q2')(s', a') − α · log π(a' | s') )       no gradient
actor loss       = mean( α · log π(a | s) − min(Q1, Q2)(s, a) )        a re-sampled, reparameterised
temperature loss = − mean( log_α · ( log π + target_entropy ).detach() )
```

- The **actor** raises Q and keeps entropy. Its gradient reaches the policy **through the
  action**, via the critic. No log-probability weighting as in days 1–2.
- **`α` is learned**, stored as `log_α`. If the entropy falls below a target (usually
  `−act_dim`), `α` rises and entropy gets more valuable; above the target, `α` falls.

### The loop

Each environment step: act, store the transitions, then one update of the critics, the actor,
the temperature and the target networks. The first steps use uniform random actions to fill the
buffer. A **ring buffer** has a fixed capacity and overwrites its oldest entries; sampling is a
random set of indices.

### Traps of the day

- No tanh correction.
- A gradient flowing through the target `y`.
- The actor loss computed with stored actions instead of freshly sampled ones.
- `τ` and `1 − τ` swapped (week 7's bug again).
- `(1 − done)` on a bool tensor: convert to float.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `squashed_sample` | reparameterised sample, tanh, corrected log-probability | the actor |
| `soft_update` | slow in-place update of a target network | the critic |
| `q_target` | the target `y`, without gradient | entropy and the temperature |
| `actor_loss`, `alpha_loss` | the two other losses | entropy and the temperature |
| `ReplayBuffer` | a ring buffer with random sampling | the loop |
| `train_sac` | the whole loop on `Reach2D` | the loop |

Run: `uv run pytest weeks/w10-reinforcement-learning/d3-sac -k "not review"`

---

## Day 4: Offline RL with IQL

Folder: `d4-offline-rl-iql`. Exercise: `w10d4_iql.py`. Provided: `offline_data.py`.

### The problem

A fixed dataset from mixed sources: good teleoperation, sloppy teleoperation, failed autonomous
rollouts. No new interaction. Two obvious methods fail, differently:

- **Behavior cloning imitates the average.** Today's data is 25% expert, 25% random and 50% an
  "anti-expert" that walks away from the goal. The average action points away from the goal.
- **Q-learning on fixed data** (yesterday's target) asks for `Q(s', a')` where `a'` is the
  policy's action, which the dataset may never contain. Nothing ever corrects those values. They
  drift upward, and the policy runs toward them. This is the **out-of-distribution action** problem.

### IQL: never ask about unseen actions

**Implicit Q-learning** trains three networks using only `(s, a, r, s')` from the dataset.

**1. `V(s)` by expectile regression.** Ordinary regression toward `Q(s, a)` over the dataset's
actions would give the **average** value of the actions seen in `s`. An **expectile** tilts it:

```
d   = Q_target(s, a) − V(s)
L_V = mean( | τ − 1[d < 0] | · d² )
```

The weight is `τ` when `V` is too low (`d > 0`) and `1 − τ` when it is too high. With `τ = 0.5`
both sides weigh the same: the mean. With `τ = 0.9`, being too low costs nine times more, so `V`
settles near the value of the **best** actions seen in that state. It approximates a maximum
without ever evaluating an action outside the data.

The two weights swapped give a `V` that tracks the **worst** actions. That is today's review bug.

```python
d = torch.tensor([1.0, -1.0])
torch.where(d < 0, torch.tensor(0.1), torch.tensor(0.9))      # the weight of each error, tau = 0.9
```

**2. `Q(s, a)` by temporal difference, with `V` for the next state:**
`y = r + γ (1 − done) V(s')`. No action is sampled at `s'`: the problem is gone.

**3. The policy by advantage-weighted regression.** Imitate the dataset's actions, each weighted
by how good it was:

```
w   = min( exp( β · (Q(s, a) − V(s)) ), w_max )          no gradient
L_π = mean( w · ||π(s) − a||² )
```

It is behavior cloning that listens mostly to the good examples. `β` sets how selective;
`w_max` stops one sample from dominating.

### Traps of the day

- Expectile weights swapped.
- A gradient through the weights `w`, or through `Q_target` in the `V` loss.
- `V(s')` used with its gradient in the Q target.
- Comparing IQL with BC on different evaluation episodes.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `expectile_loss` | the tilted squared loss | IQL, step 1 |
| `awr_weights` | clipped exponential advantage weights | IQL, step 3 |
| `train_bc` | the baseline | the problem |
| `train_iql` | the three networks and a slow Q target | IQL |

Run: `uv run pytest weeks/w10-reinforcement-learning/d4-offline-rl-iql -k "not review"`

---

## Day 5: RECAP-style advantage conditioning

Folder: `d5-recap-advantage-conditioning`. Exercise: `w10d5_recap.py`. Provided: `offline_data.py`.

### The idea

Running PPO or SAC on a flow-matching VLA with billions of parameters is not practical. **RECAP**
(the method of π*0.6) avoids policy gradients entirely. It keeps the supervised recipe of
imitation and changes what the policy is **conditioned** on:

1. Train a **value function** on all the data: demonstrations, autonomous rollouts, corrections.
2. For every step, compute an **advantage**: was this action better than usual from this state?
3. Turn it into one bit: `I = advantage > threshold`.
4. Train the policy by imitation on **all** the data, with `I` as an extra input.
5. At test time, ask for `I = 1`: "act like the better-than-usual part of the data".

No data is thrown away. Bad data is still useful: the model learns what `I = 0` looks like, and
so what to avoid. The training loss stays plain imitation.

### The n-step advantage

```
A_t = ( r_t + γ r_{t+1} + … + γ^{n−1} r_{t+n−1} ) + γⁿ V(s_{t+n}) − V(s_t)
```

Real rewards for `n` steps, then the value function for the rest, minus what was expected at
the start. Near the end of an episode fewer than `n` steps remain: sum what exists, and the value
beyond the last step is 0.

The dataset is stored flat, time-major: viewed as `(T, n_episodes)`, row `t` holds step `t` of
every episode. Reshape to that view, compute along the time axis, flatten back in the same order.

The value function here takes the time step as an input, because the horizon is fixed: the same
position is worth less with fewer steps left.

### A threshold per task

"Better than usual" depends on the task. A hard task has lower advantages everywhere. One global
threshold would label nearly all steps of the hard task `I = 0` and nearly all steps of the easy
one `I = 1`: the bit would then encode the **task**, not the quality. So the threshold is a
quantile of the advantages **within each task**.

```python
adv = torch.tensor([0.1, 0.5, 0.9, 5.0, 6.0, 7.0])
task = torch.tensor([0, 0, 0, 1, 1, 1])
torch.quantile(adv[task == 0], 0.5)          # a threshold for task 0 only
```

One threshold for everything is today's review bug.

### Conditioning with dropout

The condition has three values: `I = 0`, `I = 1`, and **dropped**. During training each sample's
condition is replaced by "dropped" with a small probability, so the policy also learns the
unconditional behaviour. It is the label dropout of classifier-free guidance (week 4, day 4).
The condition enters as a one-hot vector concatenated with the observation:

```python
F.one_hot(torch.tensor([0, 2, 1]), num_classes=3).float()
```

### Why today's data is harder than yesterday's

The sources are mixed **per step**: at every step the actor is the expert, the random policy or
the anti-expert. The state no longer reveals who is acting. Only the advantage can separate good
steps from bad ones, which is exactly what the method relies on.

### Traps of the day

- One global threshold.
- The n-step sum running past the end of the episode, or across episodes after a wrong reshape.
- Conditioning dropped for the whole batch with one coin.
- Testing with the "dropped" condition and concluding the method does nothing.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `fit_value` | regress the return-to-go on `(obs, t)` | the idea, step 1 |
| `n_step_advantage` | the formula, on the time-major view | the n-step advantage |
| `improvement_indicator` | the bit, with a per-task quantile | a threshold per task |
| `train_conditioned` | imitation with the condition as an input, sometimes dropped | conditioning with dropout |

Run: `uv run pytest weeks/w10-reinforcement-learning/d5-recap-advantage-conditioning -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one prompt from `drills/live` under a timer, AI off (`drills/live/README.md`).
- **Tick `PROGRESS.md`** and write the retro.

You are done with week 10 when you can do these without notes:

- [ ] write the return and GAE recursions, with the `done` factor
- [ ] write the REINFORCE loss and say why the baseline helps
- [ ] write the PPO clipped loss and explain the `min` case by case
- [ ] name the five pieces of SAC and what each is for
- [ ] explain why Q-learning fails offline and how the expectile avoids it
- [ ] explain RECAP in five steps and why the threshold is per task

## Optional depth (not needed for the labs)

- Interview questions on this week: `knowledge/11-reinforcement-learning.md`
- Previous week: [`../w9-mock-interviews/README.md`](../w9-mock-interviews/README.md) · Next: [`../w11-rewards-sim-sim2real/README.md`](../w11-rewards-sim-sim2real/README.md)
