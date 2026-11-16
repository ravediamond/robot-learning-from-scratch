# Week 12: The evaluation harness (your flagship engine)

**Everything you need for this week is on this page.** You need week 6 (Wilson, McNemar, paired
bootstrap, the closed-loop evaluator) and week 8 day 3 (chunk switching and delay). This week is
plain Python and the standard library, with a little torch on Friday.

Week 6 taught you to compare two policies on a toy. A real comparison of four policies on an
SO-101 is where most published robot results quietly go wrong: the session drifts (battery,
lighting, the operator learns), the operator knows which policy is running, 50 rollouts cannot
see a 10-point gap, "failure" means five different things, latency is measured on a laptop, and
nobody can rerun the exact checkpoint six months later.

This week you build the harness that fixes each of these. `flagship/` then turns it into the
real experiment: ACT, SmolVLA, VLA-JEPA and your own head on your arm, 50+ rollouts each.

## Start here

1. Read [The week in one table](#the-week-in-one-table).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-the-protocol),
   [Day 2](#day-2-comparing-several-policies), [Day 3](#day-3-failure-taxonomy),
   [Day 4](#day-4-latency-on-the-target), [Day 5](#day-5-reproducibility-the-report-a-design).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day, with its "staff angle" | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w12dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w12dN_*.py` | tests | run, never edit |
| `review_w12dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided: `evalkit.py` in every folder (Wilson, the failure taxonomy, the stage list, small
helpers). Friday adds `design.md` (you fill it) and `rubric.md` (you score yourself with it).

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w12-eval-harness/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w12-eval-harness/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w12-eval-harness/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w12-eval-harness/<day>` |

## The week in one table

| Day | Folder | The problem it fixes | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-protocol` | drift and operator bias | a counterbalanced schedule, blinding, record validation | the order policies run in |
| Tue | `d2-compare-policies` | many comparisons, small samples | Holm correction, rank probabilities, minimum detectable difference | a bootstrap that breaks pairing |
| Wed | `d3-failure-taxonomy` | "62% success" does not say what to fix | rater agreement, per-cause rates, a stage funnel | the denominator of a rate |
| Thu | `d4-latency-target` | latency measured in the wrong place, summarised the wrong way | percentiles, deadline misses, sync and async budgets | a percentile from a formula |
| Fri | `d5-repro-report` | results nobody can rerun | fingerprints, manifests, a generated table; a design | how a checkpoint is hashed |

Code on this page assumes:

```python
import hashlib, json, math, random
```

---

## Day 1: The protocol

Folder: `d1-protocol`. Exercise: `w12d1_protocol.py`.

### The one idea

The outcome of a rollout depends on the policy **and** on everything else that changes during a
session: the battery, the light through the window, the cube that wears out, the operator who
gets better at resets. Run 50 rollouts of policy A in the morning and 50 of B in the afternoon,
and you have measured the morning against the afternoon. A variable that changes together with
the thing you compare is a **confound**.

Three rules remove these confounds, and a fourth keeps the data usable.

### Rule 1: fixed start configurations

Mark 10 object positions with tape on the table. Every policy runs from every start the same
number of times. Comparisons are then **paired** (week 6, day 4): the difficulty of each start
cancels.

### Rule 2: a blocked, counterbalanced schedule

A **block** is one start configuration with every policy run once. Whatever drifts during the
session changes little inside one block, so inside a block the policies meet nearly the same
conditions.

Inside a block, somebody still goes first. If it is always the same policy, that policy always
gets the freshest reset. So the order **rotates** from block to block:

```python
order = ["a", "b", "c", "d"]
k = 1
order[k:] + order[:k]              # ['b', 'c', 'd', 'a']: the same order, rotated by one
```

Over four blocks each policy is first once, second once, and so on. That is
**counterbalancing**. Drift then hits every policy equally and cancels in the comparison.

Running all rollouts of one policy, then all of the next, is the opposite of this, and is
today's review bug. `expected_success` lets you put a number on it: give every trial a success
probability that depends only on its position in the session, and see how far apart two
identical policies land under each schedule.

A schedule must be reproducible: all randomness comes from one seeded `random.Random(seed)`.
`rng.sample(items, len(items))` returns a shuffled copy.

### Rule 3: blinding

The operator decides whether a rollout succeeded, resets the scene, and sometimes intervenes.
All three bend, without any bad intent, toward the policy they hope wins. So the run sheet shows
**aliases** ("A", "B", "C"), never names, and the mapping is opened only after the session.

### Rule 4: a record with a fixed schema, validated when written

One record per rollout, always with the same fields: trial number, alias, start, outcome, failure
tag, duration, interventions, operator. Check each record **when it is written**, not weeks
later during analysis, when the scene no longer exists and nothing can be re-run.

A validator returns the **list** of everything that is wrong (empty means valid), so the
operator can fix all of it at once: a missing key, a value of the wrong type, an outcome outside
the allowed set, a failure tag on a success. Each rule in the docstring is one check and one message.

### Traps of the day

- Policies run back to back.
- Randomness from the global `random` module instead of one seeded generator.
- A validator that stops at the first error.
- Aliases assigned in a fixed order: "A" is then always the first policy listed.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `make_schedule` | the run sheet: blocks, rotation, seeded | rules 1–2 |
| `blind_aliases` | a random policy → letter mapping | rule 3 |
| `validate_record` | a list of error messages | rule 4 |
| `expected_success` | the effect of pure drift on each policy | rule 2 |

Run: `uv run pytest weeks/w12-eval-harness/d1-protocol -k "not review"`

---

## Day 2: Comparing several policies

Folder: `d2-compare-policies`. Exercise: `w12d2_compare.py`.

### The multiple-comparisons problem

Four policies make **six** pairs. Test each pair at `α = 0.05`: each test has a 5% chance of a
false "significant" result when there is no real difference. Across six tests the chance of **at
least one** false alarm is `1 − 0.95⁶ ≈ 26%`. Run enough comparisons and something always looks
significant.

### Holm's correction

It keeps the chance of any false alarm in the whole family at 5%.

1. Sort the `m` p-values from smallest to largest.
2. Multiply the smallest by `m`, the next by `m − 1`, …, the largest by 1.
3. Make the sequence non-decreasing: each adjusted value is at least the one before it (a running maximum).
4. Cap at 1.

```python
p = [0.01, 0.04, 0.03]                      # m = 3
# sorted: 0.01, 0.03, 0.04  →  ×3, ×2, ×1  →  0.03, 0.06, 0.04  →  running max  →  0.03, 0.06, 0.06
```

Compare the adjusted values with 0.05 as usual. Holm is never less powerful than the simpler
Bonferroni correction (multiply everything by `m`).

Each pair is tested with **McNemar's exact test** on the paired outcomes (week 6, day 2).

### "Which one is best?": rank probabilities

A table of six p-values does not answer the question people ask. Answer it directly, with its
uncertainty: resample the **start configurations** with replacement, recompute every policy's
success rate on the resample, and note who is on top. Repeat. The fraction of resamples in which
a policy is on top is its probability of being the best. Ties share the credit.

**The same resampled starts for all policies.** Each resample draws one list of start indices
and applies it to every policy. That keeps the pairing: a hard start drawn twice is hard twice
for everyone. Resampling each policy separately throws the pairing away and makes every
comparison look more uncertain than it is. That is today's review bug.

### What can this sample size detect at all?

Before showing any table, say what the experiment could see. The **minimum detectable
difference** is the smallest gap that `n` rollouts per policy would detect with 80% power at
`α = 0.05`. It uses the sample-size formula of week 6 in reverse: try `d = 0.01, 0.02, …` and
keep the first one for which the needed `n` is at most yours.

With 50 rollouts per policy around 50% success, it is about 27 points. A 10-point gap needs
hundreds of rollouts per arm. Pairing helps, but the order of magnitude stands.

### Traps of the day

- Six uncorrected p-values.
- Holm without the running maximum: adjusted values that go down.
- Policies resampled independently.
- A claim of "no difference" from an experiment that could not have seen one.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `mcnemar_p` | exact two-sided test on paired outcomes | week 6 day 2 |
| `holm` | adjusted p-values | Holm's correction |
| `pairwise_table` | one row per pair: difference, discordant counts, p, adjusted p | both |
| `rank_probabilities` | probability of being the best, shared resamples | rank probabilities |
| `min_detectable_difference` | the smallest detectable gap for this `n` | what can this sample size detect |

Run: `uv run pytest weeks/w12-eval-harness/d2-compare-policies -k "not review"`

---

## Day 3: Failure taxonomy

Folder: `d3-failure-taxonomy`. Exercise: `w12d3_failures.py`.

### The one idea

"62% success" tells you which policy wins. It does not tell you what to fix. Two tools do.

### Tool 1: a failure taxonomy, and whether it can be trusted

A **taxonomy** is a short, closed list of failure causes, each with a definition, written
**before** the session: perception (wrong object or position), grasp (closes on nothing, or
drops), transport, placement, collision, timeout, safety stop. Every failed rollout gets exactly
one primary tag.

Tags are a human judgment. To know whether they mean anything, two people label the same subset
independently and you measure their agreement. Raw agreement is misleading: two raters who call
almost everything "grasp" agree most of the time by chance. **Cohen's κ** removes that chance part:

```
po = fraction of items on which the two raters agree
pe = Σ over labels of  P(rater 1 says label) · P(rater 2 says label)        agreement expected by chance
κ  = (po − pe) / (1 − pe)
```

κ = 1 is perfect agreement, 0 is what chance alone would give. Below about 0.6 the definitions
are not clear enough and the tags are noise: rewrite them before analysing anything.

```python
from collections import Counter
Counter(["grasp", "grasp", "timeout"])            # counts per label; divide by n for the probabilities
```

### Rates over all episodes

The rate of a failure cause is `count / n` where `n` is **all** the episodes of that policy that
ended in success or failure, with a Wilson interval. Not `count / number of failures`.

"Half of its failures are grasp failures" is true of a policy that fails 2 times in 100 and of
one that fails 60 times in 100. As a share of failures the number hides how often the problem
happens, and two policies cannot be compared with it. That is today's review bug. Aborted
episodes (the operator stopped for an outside reason) are excluded from `n`.

### Tool 2: the stage funnel

Cut the task into ordered stages: reach, grasp, lift, place. For each episode, record how far
it got. Then for each stage:

- **reached**: the number of episodes that completed this stage;
- **conditional**: reached, divided by the number that completed the previous stage. "Of those
  that grasped, how many lifted?"
- **cumulative**: reached, divided by all episodes.

The stage with the lowest **conditional** success is where the policy loses. The cumulative
number cannot show it: it only ever goes down.

Taxonomy and funnel together turn an evaluation into a plan: "grasp failures are most of what
separates us from ACT, they cluster on the two starts near the edge of the camera view, so the
next data collection targets those".

### Traps of the day

- Rates as a share of failures.
- κ computed, found low, and the tags analysed anyway.
- A conditional rate divided by all episodes.
- A stage nobody reached: the conditional rate has a zero denominator and must be `None`, not 0.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `cohen_kappa` | chance-corrected agreement of two raters | tool 1 |
| `breakdown` | per tag: count, rate over all episodes, Wilson bounds | rates over all episodes |
| `stage_funnel` | per stage: reached, conditional, cumulative | tool 2 |
| `biggest_drop` | the stage with the lowest conditional success | tool 2 |

Run: `uv run pytest weeks/w12-eval-harness/d3-failure-taxonomy -k "not review"`

---

## Day 4: Latency on the target

Folder: `d4-latency-target`. Exercise: `w12d4_latency.py`.

### Measuring latency properly

A policy that runs at 30 Hz on your Mac may not on the Orin. Four rules:

1. **Measure on the target**, with the real input pipeline (camera decoding, preprocessing).
2. **Drop the warm-up.** The first calls compile kernels and allocate memory; they are slower
   and not representative.
3. **On a GPU, wait for the work to finish before stopping the clock.** GPU calls return before
   the computation is done; synchronise first.
4. **Report the tail, not the mean.** A robot is hurt by its slowest steps.

A function that times something takes its **clock as an argument**. Real use passes
`time.perf_counter`; a test passes a fake clock and gets exact, repeatable numbers.

### Percentiles by nearest rank

The `q`-th percentile by **nearest rank**: sort the samples; take the one at position
`ceil(q · n) − 1` (0-based).

```python
xs = sorted([12, 15, 11, 90, 13, 14, 12, 13, 16, 12])
xs[math.ceil(0.95 * len(xs)) - 1]            # p95 = 90: the slow call is IN the answer
```

Report p50, p95, p99 and the maximum, and the **deadline-miss rate**: the fraction of calls
strictly above the budget (33.3 ms at 30 Hz).

Never estimate a percentile as "mean + 2 standard deviations". That formula assumes a bell
curve. Latency is nothing like one: most calls are fast and a few are very slow. The formula
gives a number that corresponds to no real call. That is today's review bug.

### What latency does to chunked control

Notation: `L` the inference latency, `T` the control period, `H` the chunk length, `e` the
number of actions executed from each chunk.

**Synchronous**: execute `e` actions, stop, infer, continue. The arm waits during every
inference. The idle share of the time is

```
L / ( L + e · T )
```

It shows as stutter, and it means an evaluation of a slow policy partly measured its pauses.

**Asynchronous**: start inferring the next chunk at the moment the current one starts
executing. The arm keeps moving. Two conditions must hold:

- the new chunk must arrive before the `e` actions run out: `L ≤ e · T`;
- when it arrives, its first `ceil(L / T)` actions are meant for moments already past (**stale**
  steps, week 8 day 3). After skipping them, `e` actions must remain: `ceil(L / T) + e ≤ H`.

Otherwise the robot **starves**: it has no fresh action and holds still or replays old ones. The
largest `L` that satisfies both is the latency budget of the configuration. With measured
latencies, each inference that exceeds its window costs `ceil((L − window) / T)` starved steps.

That gives the deployment sentence a research team rarely has: "on the Orin the p95 is so many
milliseconds; at 30 Hz with 50-action chunks executed 25 at a time, asynchronous inference never
starves up to this latency; synchronous execution would idle this share of the time".

### Traps of the day

- Warm-up samples kept.
- Mean and standard deviation in place of percentiles.
- "Above the budget" counted with `≥`.
- Milliseconds and seconds mixed: the clock returns seconds.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `nearest_rank` | one percentile | percentiles |
| `latency_stats` | drop warm-up, then n, p50, p95, p99, max, mean | percentiles |
| `measure` | time a function with an injected clock, in milliseconds | measuring latency properly |
| `deadline_miss_rate` | fraction strictly above the budget | percentiles |
| `sync_idle_fraction` | idle share under synchronous chunking | chunked control |
| `stale_steps`, `max_async_latency`, `starvation_steps` | the asynchronous budget | chunked control |

Run: `uv run pytest weeks/w12-eval-harness/d4-latency-target -k "not review"`

---

## Day 5: Reproducibility, the report, a design

Folder: `d5-repro-report`. Exercise: `w12d5_report.py`. Plus `design.md` and `rubric.md`.

### The one idea

A result you cannot reproduce in six months is an anecdote. Every number must be traceable to
the exact weights, data, code and settings that produced it.

### A fingerprint of the weights

A file name proves nothing: files get overwritten. A **hash** is a short code computed from
content; change one bit and the code changes. SHA-256 is the standard one.

```python
h = hashlib.sha256()
h.update("layer.weight".encode("utf-8"))       # feed bytes, piece by piece
h.update(b"\x00\x01")
h.hexdigest()[:16]                             # the fingerprint (first characters shown)
```

A fingerprint of a `state_dict` feeds, for each key **in sorted order**: the name, the dtype,
the shape, and the **raw bytes** of the tensor. Sorted, because two dicts with the same content
in a different order must give the same fingerprint. Name, dtype and shape, because the same
bytes can be read as different tensors.

Hashing `str(state_dict)` is wrong: PyTorch prints large tensors abbreviated, with `...` in the
middle. Two different checkpoints then print the same text and get the same hash. That is
today's review bug.

### A hash of the configuration

`json.dumps(config, sort_keys=True)` gives one canonical text for a dict, whatever order its
keys were written in. Hash that text.

```python
json.dumps({"b": 1, "a": 2}, sort_keys=True)          # '{"a": 2, "b": 1}'
```

### The run manifest

A small dict written **at the start** of every session: git commit, checkpoint fingerprint,
dataset version, the configuration and its hash, hardware, package versions. Stored next to the
results. If the session crashes, the manifest still says what was running.

### The manifest diff

"Why did these two runs give different numbers?" Flatten both manifests to dotted keys
(`{"a": {"b": 1}}` becomes `{"a.b": 1}`, by recursion on dict values) and list every key whose
values differ. A key present on one side only shows as `None` on the other. One command, and the
answer is a short list instead of an afternoon.

### The results table

Generated from the records, never typed by hand: a typed table drifts from the data on the
first rerun. One line per policy: `n`, the success rate with its Wilson interval, the most
frequent failure tag, the p95 latency. The exact format is in the docstring; a generated table
must be byte-for-byte stable so that a change in it means a change in the data.

### The design round

`design.md`: design an evaluation program for a generalist manipulation policy. Fill it, present
it out loud in 10 minutes, then score yourself with `rubric.md`. Everything from Monday to today
is material for it: protocol, sample sizes and what they can detect, failure analysis, latency,
reproducibility.

### Traps of the day

- A checkpoint identified by its file name, or hashed through `str()`.
- Keys hashed in insertion order.
- A manifest written at the end of the session.
- A results table edited by hand "just once".

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `tensor_fingerprint` | SHA-256 over sorted names, dtypes, shapes, raw bytes | a fingerprint of the weights |
| `config_hash` | hash of the canonical JSON | a hash of the configuration |
| `run_manifest` | the dict of everything that defines a run | the run manifest |
| `flatten`, `manifest_diff` | dotted keys; what differs between two runs | the manifest diff |
| `results_table` | the Markdown table, from records | the results table |

Run: `uv run pytest weeks/w12-eval-harness/d5-repro-report -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one prompt from `drills/live`, timer on, AI off.
- **Tick `PROGRESS.md`** and write the retro.
- **Then:** `flagship/README.md` turns this week's functions into the real four-policy comparison.

You are done with week 12 when you can say, in one breath each:

- [ ] how you control drift and operator bias: fixed starts, blocked, counterbalanced, blinded, validated logs
- [ ] why six comparisons need a correction, and how Holm works
- [ ] what your sample size can and cannot detect
- [ ] why failure rates are over all episodes, and what κ is for
- [ ] p50, p95, p99 by nearest rank; the two conditions of asynchronous chunking
- [ ] what is in a run manifest and how a checkpoint is fingerprinted

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/13-evaluation-at-scale.md`
- The real experiment: `flagship/PROTOCOL.md`, `flagship/PREREG.md`, `flagship/RUNBOOK.md`
- Previous week: [`../w11-rewards-sim-sim2real/README.md`](../w11-rewards-sim-sim2real/README.md) · Next: [`../w13-robot-data/README.md`](../w13-robot-data/README.md)
