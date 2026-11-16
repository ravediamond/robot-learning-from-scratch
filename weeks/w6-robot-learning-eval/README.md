# Week 6: Robot learning and evaluation

**Everything you need for this week is on this page.** You need week 1 (behavior cloning), week 4
(VAE, flow policy, closed-loop rollout). Nothing from other folders is required.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [The idea of the week](#the-idea-of-the-week).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-act-and-temporal-ensembling),
   [Day 2](#day-2-success-rate-statistics), [Day 3](#day-3-failure-detection),
   [Day 4](#day-4-a-closed-loop-evaluator), [Day 5](#day-5-compounding-error-and-dagger).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w6dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w6dN_*.py` | tests | run, never edit |
| `review_w6dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided helpers: `sim2d.py` (week 4's obstacle world), `toy2d.py`, `policies_ref.py` (week 4's
two policies, so this week does not depend on your own), `dagger_world.py` (day 5),
`length_null_real.py` (day 3, for your own rollouts).

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve (Monday: 10-min delayed recall instead) | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w6-robot-learning-eval/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w6-robot-learning-eval/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w6-robot-learning-eval/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w6-robot-learning-eval/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-act` | ACT: a conditional VAE over chunks; temporal ensembling | the policy, its loss, the ensembler | stale actions in the ensembler |
| Tue | `d2-success-statistics` | intervals, paired tests, sample sizes | Wilson, bootstrap, McNemar, trials needed | resampling the wrong unit |
| Wed | `d3-failure-detection` | AUROC, operating points, conformal thresholds | the four metrics of a failure monitor | ties |
| Thu | `d4-closed-loop-eval` | what a real evaluator does | an evaluator with fixed, paired starts | starts that are not fixed |
| Fri | `d5-dagger` | compounding error, DAgger | BC against DAgger under disturbances | who labels the new data |

## The idea of the week

Two questions decide whether a robot-learning result is worth anything:

1. **Does the policy work on the robot?** Measured in closed loop, never by a validation loss.
2. **How sure are you?** 14 successes out of 20 against 11 out of 20 is, on its own, no evidence
   that the first policy is better.

Most of this week is the second question. It is what separates a result from an anecdote, and it
is the part of your own work (the register, the failure-monitor paper) that interviewers will
push on.

All code on this page assumes:

```python
import math
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Day 1: ACT and temporal ensembling

Folder: `d1-act`. Exercise: `w6d1_act.py`. Provided: `sim2d.py`, `toy2d.py`, `policies_ref.py`.

### ACT in one paragraph

ACT (Action Chunking with Transformers, Zhao et al. 2023) is the classic policy for small data,
about 50 demonstrations, and a LeRobot default for the SO-101. It is a **conditional VAE** over
action chunks. Two ideas: predict a chunk, not one action; and give the model a latent `z` to
absorb what the observation cannot explain, such as which way this operator went around.

### The model

It is the VAE of week 4, day 1, with the observation as a condition on both sides. Today's
version uses MLPs where the original uses transformers.

```
training:   encoder(obs, true chunk) → mu, logvar
            z = mu + std · eps
            decoder(obs, z) → predicted chunk
            loss = masked L1(predicted chunk, true chunk) + beta · KL( q(z | obs, chunk) || N(0, I) )
test time:  z = 0, the mean of the prior
            decoder(obs, 0) → chunk
```

In training the encoder **sees the answer**, the true chunk, and compresses into `z` whatever
helps the decoder reconstruct it. The KL keeps `z` close to `N(0, I)` so that little information
passes; `beta` sets how little (ACT uses a large `beta`). At test time there is no true chunk:
`z` is set to 0 and the policy is deterministic.

### L1, not MSE

MSE's best constant is the **mean** (week 4). L1's best constant is the **median**: the value
with half of the data on each side. On a bimodal target the median sits inside one of the two
groups, or anywhere between them when the groups are equal in size. So an L1 policy tends to pick
a side where an MSE policy averages. Keep that in mind when you read today's collision numbers.

The masked L1 is the masked mean of week 1 on `|pred − target|`, over real `(step, dim)` entries.

### Temporal ensembling

Week 4 executed the first `H_exec` actions of a chunk, then replanned. At each replan the new
chunk may disagree with the old one, and the arm jerks.

ACT's alternative: query the policy at **every** step. At time `t`, the current action was
predicted by several chunks: the one made now (its action 0), the one made one step ago (its
action 1), and so on, up to `H` of them. Execute their weighted average.

```
weights  w_i = exp(−m · i),   i = 0 for the OLDEST prediction, normalised to sum to 1
```

In ACT's convention older predictions weigh more, which gives smoother motion. `m` sets how
fast the weights fall.

```python
w = torch.exp(-0.1 * torch.arange(4.0))        # 4 predictions available, oldest first
w / w.sum()                                    # normalised
```

**The bookkeeping** is the whole exercise. Keep the recent chunks in a buffer. At each call, a
chunk that was made `a` steps ago contributes its action number `a`. A chunk made `H` or more
steps ago has no action left for the current step and must be dropped. During the first steps
of an episode there are fewer than `H` chunks: the average is over what exists. Using an action
at the wrong offset, or keeping a chunk that has run out, is today's review bug.

The price is one inference per control step, against one every `H_exec` steps.

### Traps of the day

- Sampling `z` at test time when the policy should use `z = 0`.
- Reconstruction as a mean and KL as a sum.
- Ensembler weights not normalised, or indexed newest first.
- An ensembler that is not reset between episodes.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `ACTPolicy` | `encode`, `decode`, `forward`, `act` with `z = 0` or a sampled `z` | the model |
| `act_loss` | masked L1 + `beta` · mean KL | L1, not MSE; week 4 day 1 |
| `train_act` | the loop | weeks 1–2 |
| `TemporalEnsembler` | one `step(chunk)` per control step → the action to execute | temporal ensembling |

Run: `uv run pytest weeks/w6-robot-learning-eval/d1-act -k "not review"`

---

## Day 2: Success-rate statistics

Folder: `d2-success-statistics`. Exercise: `w6d2_stats.py`. torch and the standard library only.

### Why this matters

A success rate from 20 trials is one draw of a random number. Run the same policy again and you
get another one. Statistics answers: how far from the truth can my number be, and is the
difference between two policies real?

### Tool 1: the Wilson interval

`k` successes in `n` trials. `p = k/n` is the estimate. A **95% confidence interval** is a range
built so that, in 95% of experiments, it contains the true rate.

```
z = 1.96                                    95%
d      = 1 + z²/n
centre = ( p + z²/(2n) ) / d
half   = z · sqrt( p(1−p)/n + z²/(4n²) ) / d
interval = ( centre − half, centre + half )
```

7 out of 10 gives (0.40, 0.89). 70 out of 100 gives (0.60, 0.78). Halving the width costs four
times the trials.

Use Wilson, not the textbook `p ± 1.96 · sqrt(p(1−p)/n)`. That simpler formula fails exactly
where robotics lives: small `n`, and rates near 0 or 1. With 10 out of 10 it returns the
interval (1, 1), "certainly perfect", which is absurd.

### Tool 2: the bootstrap

For a statistic with no formula (a mean return, a difference, an AUROC), simulate the repetition
of the experiment using the data you have:

1. Draw `n` items from your `n` items **with replacement** (some appear twice, some not at all).
2. Compute the statistic on that resample.
3. Repeat `B` times (1000 or more).
4. The 2.5th and 97.5th percentiles of the `B` values are the interval.

```python
g = torch.Generator().manual_seed(0)
vals = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])       # one outcome per episode
idx = torch.randint(0, 6, (1000, 6), generator=g)          # 1000 resamples of 6 indices
means = vals[idx].mean(dim=1)                              # the statistic on each resample
torch.quantile(means, torch.tensor([0.025, 0.975]))        # the interval
```

**The unit you resample is the unit that is independent: the episode, never the frame.** Frames
of one episode move together. Resampling frames pretends you have thousands of independent
observations and gives an interval that is far too narrow. This is the same idea as the episode
split of week 1, and today's review bug.

### Tool 3: paired comparisons

To compare policies A and B, run both from the **same** start states. Some starts are hard for
everyone, some easy for everyone; pairing cancels that shared difficulty and leaves the
difference between policies.

**Paired bootstrap.** Compute the difference per start, `a − b`, and bootstrap its mean,
resampling start states. If the interval excludes 0, the difference is unlikely to be chance.

**McNemar's exact test**, for success/failure outcomes. Sort the starts in four groups:

| | B succeeds | B fails |
|---|---|---|
| **A succeeds** | both | `n10`: A only |
| **A fails** | `n01`: B only | neither |

Starts where both agree say nothing about which is better. Only the **discordant** pairs
matter. If the two policies were equally good, each discordant pair would be a fair coin flip
between "A only" and "B only". The p-value is the probability, under that fair coin, of a split
at least as lopsided as the one observed:

```
m = n01 + n10
p = min( 1, 2 · P( Binomial(m, 0.5) ≤ min(n01, n10) ) )
P( Binomial(m, 0.5) ≤ j ) = Σ_{i=0..j} C(m, i) / 2^m          math.comb(m, i)
```

A **p-value** is the probability of seeing data this extreme if there were no real difference.
Small (below 0.05 by convention) means chance is an unlikely explanation.

### Tool 4: how many trials

Before an experiment: how many trials per policy to tell a rate `p1` from a rate `p2`? Two
settings: `α`, the accepted risk of announcing a difference that is not there (0.05), and
**power** `1 − β`, the chance of detecting a difference that is there (0.8).

```
p̄ = (p1 + p2)/2,   q = 1 − p
n per arm = ( z_{1−α/2} · sqrt(2 p̄ q̄) + z_{1−β} · sqrt(p1 q1 + p2 q2) )² / (p1 − p2)²
```

```python
statistics.NormalDist().inv_cdf(0.975)        # 1.96: the z for a two-sided 5%
statistics.NormalDist().inv_cdf(0.80)         # 0.84: the z for 80% power
```

The difference is squared in the denominator: telling 70% from 80% takes about 290 trials per
arm; telling 70% from 90%, about 60. This is for two **independent** groups; pairing needs fewer.
It is why 20 trials per policy can only show very large effects.

### Traps of the day

- `p ± 1.96·se` at small `n`.
- Bootstrap over frames.
- Two policies compared on different start states and then treated as paired.
- A sample size rounded down.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `wilson` | the interval, clipped to [0, 1] | tool 1 |
| `bootstrap_ci` | episode bootstrap for any statistic | tool 2 |
| `paired_bootstrap_diff` | mean difference and its interval | tool 3 |
| `mcnemar_exact` | two-sided exact p-value from the discordant counts | tool 3 |
| `trials_needed` | trials per arm, rounded up | tool 4 |

Run: `uv run pytest weeks/w6-robot-learning-eval/d2-success-statistics -k "not review"`

---

## Day 3: Failure detection

Folder: `d3-failure-detection`. Exercise: `w6d3_monitors.py`. Provided: `length_null_real.py`.

### A monitor is a score and a threshold

A **failure monitor** watches an episode and outputs a score: higher means "more likely to be
failing". Above a threshold, it raises an alarm. Convention this week: label 1 = failure (the
positive class), 0 = success.

Two questions: how good is the score, whatever the threshold? And what happens at the threshold
you actually deploy?

### AUROC: the quality of the score

**AUROC** is the probability that a randomly chosen failure has a higher score than a randomly
chosen success. Ties count for one half. 1.0 is perfect separation, 0.5 is a coin.

The direct computation compares every failure with every success, with the grid trick:

```python
pos = torch.tensor([0.9, 0.4])                # scores of the failures
neg = torch.tensor([0.1, 0.4, 0.3])           # scores of the successes
gt = (pos[:, None] > neg[None, :]).float()    # (n1, n0): 1 where the failure scores higher
eq = (pos[:, None] == neg[None, :]).float()
(gt + 0.5 * eq).mean()                        # tensor(0.9167): 5.5 of the 6 pairs
```

The **rank form** (Mann–Whitney) gives the same number with a sort instead of a grid. Rank all
scores together from 1 (smallest) to `n`; then

```
AUROC = ( sum of the ranks of the failures − n1·(n1 + 1)/2 ) / ( n1 · n0 )
```

**Ties must get their average rank.** Three scores tied for ranks 4, 5 and 6 all get rank 5. A
plain sort gives tied scores ranks in whatever order they sit in the array, and then the AUROC
depends on the order of the rows. Monitors that output a few discrete values (a step count, a
rounded score) tie constantly. This is today's review bug, and the direct form above is your check.

### An interval for the AUROC

Bootstrap over episodes (tool 2 of yesterday): resample (score, label) rows together, recompute.
A resample that happens to contain only one class has no AUROC; skip it.

### The operating point

You deploy a threshold, not an AUROC. The usual question: "I accept false alarms on 5% of good
episodes. What fraction of failures do I catch?"

- **FPR** (false positive rate): fraction of successes flagged.
- **TPR** (true positive rate): fraction of failures flagged.

TPR at FPR = 5%: set the threshold at the 95th percentile of the **success** scores
(`torch.quantile(success_scores, 0.95)`), then count the fraction of failures above it. Two
monitors with the same AUROC can differ a lot here.

### A conformal threshold

A way to choose the threshold with a **guarantee** on false alarms and no assumption about the
score's distribution. Take `n` success episodes kept aside for calibration, and sort their scores.

```
k = ceil( (n + 1) · (1 − α) )
threshold = the k-th smallest calibration score          (1-based)
```

Then on a new success episode, the probability that its score exceeds the threshold is at most
`α`. The only assumption is that the new episode comes from the same conditions as the
calibration ones (exchangeability). The `n + 1` accounts for the new episode itself. If `k > n`
there are too few calibration episodes for that `α`: no finite threshold gives the guarantee.

### The cheapest baseline: episode length

Failed episodes tend to run long: the robot tries, retries, times out. So "elapsed time" is
already a failure score, and it costs nothing. Your own paper found this null hard to beat.
Any monitor must beat it **on the same episodes, paired**, before its sophistication means anything.

After the tests pass, on your Mac:

```bash
uv run python weeks/w6-robot-learning-eval/d3-failure-detection/length_null_real.py
```

It computes the length null on your own rollouts with **your** functions.

### Traps of the day

- Ties ranked by position.
- The threshold computed on all scores instead of the success scores.
- A 0-based index for the `k`-th smallest.
- Labels with the opposite meaning (1 = success): the AUROC becomes `1 − AUROC`.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `auroc` | rank form with average ranks | AUROC |
| `auroc_ci` | episode bootstrap, single-class resamples skipped | an interval |
| `tpr_at_fpr` | threshold from the success scores, fraction of failures above | the operating point |
| `conformal_threshold` | the `k`-th smallest calibration score, or infinity | a conformal threshold |

Run: `uv run pytest weeks/w6-robot-learning-eval/d3-failure-detection -k "not review"`

---

## Day 4: A closed-loop evaluator

Folder: `d4-closed-loop-eval`. Exercise: `w6d4_eval.py`. Provided: `sim2d.py`, `toy2d.py`,
`policies_ref.py`.

### Why this matters

From today, every policy is judged by an evaluator, not by its loss. An evaluator is an
instrument, and an instrument that is not the same for everyone measures nothing.

### What a real evaluator does

1. **Fixed start states.** Generated once from a seed and reused for every policy. Comparisons
   are then **paired** (day 2): the difficulty of the starts cancels. An evaluator that draws
   new starts at each call gives two policies two different exams. Today's review bug.
2. **Per-episode outcomes**, not just a rate: for each start, success, collision, number of
   steps. You need them for the bootstrap, for McNemar, and to look at the failures.
3. **The same execution settings** for everyone: `H_exec`, `T_MAX`, clipping.
4. **Reports with intervals** and the paired table: `n11` both succeed, `n10` A only, `n01` B
   only, `n00` neither.

The rollout itself is week 4, day 5, with two changes: the starts are **given** (and must not be
modified: work on a copy, `starts.clone()`), and the function returns one outcome per episode.
The policy's own randomness (the flow head samples noise) comes from a second seeded generator.

Counting pairs from two boolean tensors:

```python
a = torch.tensor([True, True, False, False])
b = torch.tensor([True, False, True, False])
int((a & ~b).sum())                    # A succeeds and B fails
```

### The experiment: validation loss against closed loop

Train week 4's MSE head and flow head on the same bimodal demonstrations. Measure two things:

- **(a)** validation chunk MSE on held-out demonstration episodes;
- **(b)** closed-loop success on 200 fixed starts.

Write down which policy you expect to rank first on each, **before** running.

Think about what (a) rewards. For one observation, the held-out demonstration went above or
below the obstacle. A policy that outputs the average of the two is at a medium distance from
both. A policy that commits to one side is right half of the time and far off the other half.
Which has the lower squared error? And which one crosses the obstacle?

That is why "validation loss went down" is not evidence that a policy improved.

### Traps of the day

- Starts regenerated inside `evaluate`, or modified in place.
- Outcomes summarised to a rate too early.
- Two policies evaluated with different `H_exec`.
- Validation MSE computed over padded steps.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `fixed_starts` | seeded starts, same seed same starts | what a real evaluator does |
| `evaluate` | rollout from given starts → per-episode success, collision, steps | what a real evaluator does; week 4 day 5 |
| `paired_counts` | the four counts | what a real evaluator does |
| `val_chunk_mse` | open-loop error on held-out demonstrations | the experiment |

Run: `uv run pytest weeks/w6-robot-learning-eval/d4-closed-loop-eval -k "not review" -s`

---

## Day 5: Compounding error and DAgger

Folder: `d5-dagger`. Exercise: `w6d5_dagger.py`. Provided: `dagger_world.py` (read it first), `sim2d.py`.

### The problem

Behavior cloning learns from the states the **expert** visited. The expert is good, so those
states lie on a narrow path. In closed loop, a small error or an outside disturbance puts the
robot slightly off that path, in a state it has never seen. There its prediction is worse; the
next state is further off; and so on. Errors **compound**. In the worst case the cost of BC
grows with the square of the episode length.

More demonstrations of the same kind do not fix it: they add more data on the path, none off it.

### DAgger (dataset aggregation)

Collect data where the **policy** goes, labelled with what the **expert** would do there.

```
D = expert demonstrations;  π = train(D)
repeat:
    run π, with the disturbances it will face, and record the states it visits
    label those states with the EXPERT's action
    D = D ∪ { (state, expert(state)) };  π = train(D)
```

The line that matters is the labelling. The states come from the policy; the actions come from
the expert. Recording the **policy's own** actions on those states would teach the policy to
repeat its mistakes: it would learn nothing new. That is today's review bug.

After one or two rounds the dataset contains off-path states with the correct recovery action,
and the policy learns to come back.

**On a real robot** the expert cannot label states after the fact. The practical forms are a
human who takes over when the policy drifts (interventions, HG-DAgger), and recovery data
collected on purpose: push the arm off course during teleoperation and record the correction.

### The setting today

`dagger_world.py` adds a push to the obstacle world: at step `KICK_T`, each robot is kicked off
its path. Policies are single-step (one action per observation, no chunk), trained on
`action / MAX_STEP` as in week 4.

The rollout is the one you know, with the push added at the right step, and an option to record
the positions of the episodes that are still running. The docstring fixes the order of
operations inside a step and which generator draws what; follow it exactly, because the
comparison between BC and DAgger is **paired on the same 200 starts and pushes**.

The test: BC on 10 demonstrations stays below 70% success under pushes; DAgger after 2
iterations exceeds 90%.

### Traps of the day

- New states labelled with the policy's actions.
- Rollouts for data collection run without the disturbance.
- The dataset replaced by the new states instead of extended with them.
- BC and DAgger evaluated on different starts.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `train_bc` | single-step MLP policy, MSE | week 1 day 5 |
| `run` | rollouts with the push, optional recording of visited states | the setting today |
| `aggregate` | append visited states with the expert's labels | DAgger |
| `dagger` | the loop: run, aggregate, retrain | DAgger |

Run: `uv run pytest weeks/w6-robot-learning-eval/d5-dagger -k "not review" -s`

---

## End of the week

- **AI-off gate (15 min):** one drill from `drills/w6` (dynamic programming), timer on, AI off.
- **Reading slot (20 min, read only):** `reading/README.md`, row 6 (your own published analysis: unit of resampling, strata, ties).
- **Tick `PROGRESS.md`**, write the retro and the hardest lab.

You are done with week 6 when you can do these without notes:

- [ ] draw ACT at training time and at test time, and say why L1
- [ ] write the Wilson interval and say why not `p ± 1.96·se`
- [ ] explain the bootstrap and why the episode is the unit
- [ ] explain McNemar's test with the discordant pairs
- [ ] compute an AUROC by hand on five scores, with a tie
- [ ] list the four properties of a real evaluator
- [ ] explain compounding error and the one line that makes DAgger work

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/07-robot-eval-statistics.md`
- Uncertainty, at length, with your own numbers: `reference/journal/book/07-uncertainty.md`
- Your failure-monitor paper: `reference/journal/papers/failure-monitors/README.md`
- Previous week: [`../w5-vlm-vla/README.md`](../w5-vlm-vla/README.md) · Next: [`../w7-jepa-world-models/README.md`](../w7-jepa-world-models/README.md)
