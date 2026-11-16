# Week 13: Robot data engineering

**Everything you need for this week is on this page.** You need week 2 day 4 (reading a LeRobot
dataset: gaps, idle starts, lag), week 6 (episode splits, intervals, DAgger) and week 12 day 5
(hashing). This week is numpy and the standard library.

In robot learning the data is the model's ceiling, and it is collected by tired humans on
hardware that drifts. Research teams now hire for this explicitly ("Research Engineer – Data"):
someone who can say what is in 10 000 hours of teleoperation, what should not be, what to
collect next, and prove that a change of mixture helped. This week builds that toolkit.

## Start here

1. Read [The week in one table](#the-week-in-one-table).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-dataset-qa-at-scale),
   [Day 2](#day-2-curation-splits-versioning), [Day 3](#day-3-mixtures-scaling-ablations),
   [Day 4](#day-4-video-and-storage-at-scale), [Day 5](#day-5-the-data-flywheel-a-design).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day, with its "staff angle" | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w13dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w13dN_*.py` | tests | run, never edit |
| `review_w13dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided: `d1-dataset-qa/fake_fleet.py` (episodes with known defects). Friday adds `design.md`
(you fill it) and `rubric.md`.

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w13-robot-data/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w13-robot-data/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w13-robot-data/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w13-robot-data/<day>` |

## The week in one table

| Day | Folder | The question | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-dataset-qa` | which episodes should not be trained on? | automatic checks and a quarantine report | how outliers are found |
| Tue | `d2-curation-splits` | is my split honest, is my dataset versioned? | near-duplicate search, group splits, a fingerprint, a ledger | how the split is drawn |
| Wed | `d3-mixtures-scaling` | how do I mix sources, and did a change help? | sampling weights, a scaling fit, a paired ablation | one division |
| Thu | `d4-video-at-scale` | will the loader keep up? | storage and decode budgets, shard assignment, a shuffle buffer | who reads which shard |
| Fri | `d5-flywheel` | what do I do with the policy's own rollouts? | intervention extraction, a review queue, a collection plan; a design | which actions become labels |

Code on this page assumes:

```python
import hashlib, math, random
import numpy as np
```

---

## Day 1: Dataset QA at scale

Folder: `d1-dataset-qa`. Exercise: `w13d1_qa.py`. Provided: `fake_fleet.py`.

### The one idea

A dataset of 2 000 episodes always contains episodes that teach the wrong thing: a camera that
froze for eight seconds, a joint whose encoder died, an arm driven into its limit, the same
episode uploaded twice, a three-second "episode" that was a mis-click. Nobody watches 2 000
videos. You write **checks**, run them on every upload, and **quarantine** each bad episode with
a written reason. In week 2 you checked one episode by hand; this is the same job as a system.

### The checks

| Check | What it looks at |
|---|---|
| frozen camera | the longest run of identical consecutive frames |
| dead joint | a joint whose range over the episode is about zero while other joints move |
| limit hits | the fraction of frames where some joint sits at, or beyond, its calibrated limit |
| length outliers | episodes far shorter or longer than usual |
| duplicates | two episodes with identical state and action arrays |

**Frozen camera.** Comparing images pixel by pixel across a dataset is slow; compare a **hash**
of each frame. The check is then "the longest run of equal consecutive values" in a list: walk
it once, keep the current run length and the best seen.

**Dead joint.** `x.max(axis=0) − x.min(axis=0)` gives each joint's range over the episode. A
joint is dead only if **others** moved: an arm that did not move at all is a different problem
(an idle episode) and should not be reported as six dead joints.

**Limit hits.** Limits are a `(J, 2)` array of `[low, high]`. A frame counts if **any** joint is
within a margin of a limit: build a boolean `(T, J)` array, reduce with `.any(axis=1)`, take the mean.

### Outliers: median and MAD, not mean and σ

The textbook rule flags values beyond `mean ± 3σ`. It fails at exactly this job. One 40-minute
episode among 2-minute ones pulls the mean up and inflates σ so much that the outlier is no
longer three σ away. It hides itself, and everything else with it. This is called **masking**.

The robust version replaces the mean by the **median** and σ by the **median absolute deviation**:

```
MAD      = median( | x − median(x) | )
robust z = 0.6745 · ( x − median(x) ) / MAD
```

Neither the median nor the MAD moves when a few values are extreme. The constant 0.6745 makes
the robust z comparable to an ordinary z-score on well-behaved data. If the MAD is 0 (more than
half the values identical), the score is undefined: return zeros.

```python
x = np.array([2.0, 2.1, 1.9, 2.0, 40.0])
(x - x.mean()) / x.std()                 # the outlier scores only 2.0: not flagged at 3
med = np.median(x); mad = np.median(np.abs(x - med))
0.6745 * (x - med) / mad                 # the outlier scores above 250
```

Mean and σ in the outlier check is today's review bug.

### Duplicates: a content hash

Two uploads of the same recording have different ids, timestamps and station names, and the
same data. Hash **only the content**: the state and action arrays, converted to a fixed dtype
and memory order so that equal numbers give equal bytes, plus their shapes.

```python
a = np.arange(6, dtype=np.float32).reshape(2, 3)
h = hashlib.sha256()
h.update(str(a.shape).encode())
h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
h.hexdigest()[:12]
```

Group episodes by hash; in each group keep one and report the others as duplicates of it.

### The report

One function runs every check and returns, per quarantined episode, the sorted list of
**reasons**, plus the list of duplicate pairs. Reasons are fixed strings, so they can be counted:
"this week 4% of uploads from station 3 have a frozen camera" is how hardware faults are found.

### Traps of the day

- Mean ± 3σ.
- A motionless episode reported as dead joints.
- Metadata inside the content hash: duplicates no longer collide.
- Frozen-frame thresholds in frames where the docstring gives seconds: convert with the fps.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `longest_frozen_run` | longest run of identical consecutive values | the checks |
| `dead_joints` | joints that do not move while others do | the checks |
| `limit_fraction` | share of frames at or beyond a limit | the checks |
| `robust_z` | median and MAD z-score | outliers |
| `content_hash` | SHA-256 of state and action content | duplicates |
| `qa_report` | all checks → quarantine reasons and duplicate pairs | the report |

Run: `uv run pytest weeks/w13-robot-data/d1-dataset-qa -k "not review"`

---

## Day 2: Curation, splits, versioning

Folder: `d2-curation-splits`. Exercise: `w13d2_curation.py`.

Three ways a dataset still lies to you after QA.

### Near-duplicates

An operator records the same motion 30 times from the same start. The files differ (noise,
slightly different lengths), so yesterday's hash does not catch them. They inflate the dataset
without adding information, and in a random split, copies of one demonstration land on both sides.

To compare two trajectories of different lengths, **resample** both to the same number of
points, evenly spaced in normalised time from 0 to 1, by linear interpolation:

```python
traj = np.array([0.0, 1.0, 4.0])                       # 3 points over normalised time 0, 0.5, 1
np.interp(np.linspace(0, 1, 5), np.linspace(0, 1, 3), traj)     # 5 points: [0, 0.5, 1, 2.5, 4]
```

`np.interp` works on one dimension; apply it to each joint. Two resampled trajectories are
near-duplicates if their **RMS difference** (the square root of the mean squared difference,
over points and dimensions) is below a tolerance.

### Leaky splits

Week 1: split by episode, not by frame. At scale that is not enough. Episodes from the same
**session** share lighting, objects, operator habits and often the same starts. With a random
split of episodes, validation measures how well the model memorised the session.

Split by **group**: session, station, operator or day. Every episode of a group goes to the same
side. Shuffle the groups with a seed, and move whole groups to validation until it is large enough.

A random episode split is today's review bug. The check to add to any split:
`set(train_groups) & set(val_groups)` must be empty.

### Untraceable versions

"The dataset" changes every week: new uploads, removals. A training run must say exactly which
data it used.

- **A dataset fingerprint**: hash the **sorted** list of per-episode content hashes. Sorted, so
  the order of files does not matter; any added, removed or modified episode changes it.
- **A removal ledger**: an append-only list of every episode dropped, by which rule, in which
  version. Append-only means entries are never edited or deleted. It makes every curation
  decision reviewable and reversible: "restore everything removed by the frozen-camera rule in v7".

### Traps of the day

- Near-duplicates compared without resampling: different lengths cannot be subtracted.
- A split by episode when sessions exist.
- A fingerprint that depends on file order.
- Episodes deleted with no record of why.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `resample` | `(T, D)` → `(n, D)` by linear interpolation | near-duplicates |
| `near_duplicates` | pairs with a small RMS difference | near-duplicates |
| `group_split` | whole groups to train or validation, seeded | leaky splits |
| `dataset_fingerprint` | hash of the sorted episode hashes | untraceable versions |
| `Ledger` | append-only record of removals | untraceable versions |

Run: `uv run pytest weeks/w13-robot-data/d2-curation-splits -k "not review"`

---

## Day 3: Mixtures, scaling, ablations

Folder: `d3-mixtures-scaling`. Exercise: `w13d3_mixtures.py`.

### The one idea

A modern robot policy trains on a **mixture**: your 50 hours, a public 1 000-hour dataset,
simulation, other robots. How much of each the model sees matters as much as the architecture.

### Temperature sampling

Sample source `i` with probability proportional to `n_i^(1/T)`, where `n_i` is its size.

| `T` | Effect |
|---|---|
| 1 | proportional to size: the large source drowns yours |
| very large | uniform over sources: your 50 hours count as much as the 1 000 |
| 2 to 5 | the usual compromise |

```python
sizes = np.array([50.0, 1000.0])
w = sizes ** (1 / 2.0)                    # T = 2
w / w.sum()                               # about [0.18, 0.82], against [0.05, 0.95] at T = 1
```

### From source weights to item weights

A sampler draws **items**, not sources. For source `s` to be picked with probability `p_s`,
each of its items needs the weight `p_s / (number of items in s)`. The items of a source share
its probability.

Giving every item of `s` the weight `p_s` itself multiplies each source by its size again: the
large source gets even more than proportional. That missing division is today's review bug.

### Epochs per source

With a budget of `B` samples, source `i` is seen `B · p_i / n_i` times. A small source with a
large weight may be passed over dozens of times while the large one is not seen once in full.
Repetition is a risk of overfitting on the small source; compute the number before training.

### Scaling: what more data buys

Train on a few dataset sizes, measure the error, and fit a **power law**:

```
error ≈ a · N^(−b)        ⇔        log(error) = log(a) − b · log(N)
```

In log-log coordinates it is a straight line, so the fit is ordinary least squares on the logs:

```python
n = np.array([100.0, 200.0, 400.0]); err = np.array([0.40, 0.28, 0.20])
slope, intercept = np.polyfit(np.log(n), np.log(err), 1)
-slope, np.exp(intercept)                 # b ≈ 0.5, a ≈ 4
```

`b` says what doubling the data gives: the error is multiplied by `2^(−b)`. With `b = 0.5`,
about 0.71. Extrapolate a little, never a lot: the law holds for more data **of the same kind**.

### Ablations, paired by seed

"Removing source X changes success by −4 points." Training twice with different seeds also
changes success, by several points. Without a measure of that noise, the −4 means nothing.

Run the ablation over several seeds, **the same seeds with and without X**. For each seed take
the difference `d_i`. Then:

```
effect = mean(d)          se = sd(d) / sqrt(k)          interval = effect ± z · se
```

`sd` with `ddof = 1` (the sample standard deviation). Pairing by seed removes the part of the
noise that both runs share (the initialisation, the data order), exactly as pairing by start
state did in week 6. The exercise also returns the unpaired standard error, so you can see how
much pairing gained.

### Traps of the day

- Item weights not divided by the size of their source.
- Temperature applied as `n^T`.
- A power law fitted on raw values instead of logs.
- An ablation with one seed.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `temperature_weights` | source probabilities from sizes and `T` | temperature sampling |
| `item_weights` | per-item weights that realise the source probabilities | from source weights to item weights |
| `epochs_per_source` | expected passes over each source | epochs per source |
| `fit_power_law` | `a`, `b` from sizes and errors | scaling |
| `ablation_effect` | paired effect, standard error, interval | ablations |

Run: `uv run pytest weeks/w13-robot-data/d3-mixtures-scaling -k "not review"`

---

## Day 4: Video and storage at scale

Folder: `d4-video-at-scale`. Exercise: `w13d4_video.py`.

Robot datasets are mostly video. Three numbers decide whether training is fed or starved.

### Storage

A raw frame is `width × height × 3` bytes. At 640×480 and 30 fps:

```
640 · 480 · 3 · 30 = 27 648 000 bytes per second ≈ 27.6 MB/s per camera ≈ 100 GB per hour
```

Compressed video (H.264, AV1) is 100 to 500 times smaller. Video bitrates are given in
**megabits** per second: divide by 8 for bytes. A terabyte is 10¹² bytes. Know both numbers
before promising "we keep everything".

### The cost of a random frame

Compression stores most frames as differences from earlier frames. Only **keyframes** are
complete images. Frames come in groups (a **GOP**, group of pictures), each starting with a
keyframe. To decode frame `k` the decoder starts at the previous keyframe and decodes every
frame up to `k`.

With a keyframe every `g` frames, a frame at a random position in its group costs on average
`(g + 1) / 2` decoded frames. With `g = 60`, about 30 frames decoded to obtain one.

A training loader that samples random frames from thousands of files is therefore limited by
decoding, not by the GPU. The fixes: shorter GOPs (larger files), sampling **contiguous
windows** (one seek, many useful frames), or caching decoded frames, as you did in week 2.

The budget: `samples per second × decoded frames per sample`, divided by what one CPU core
decodes per second, rounded **up**, gives the cores the loader needs.

### Sharding

A large dataset is stored as many files, the **shards**. Training runs several processes
(ranks, week 9), each with several loader workers. Every `(rank, worker)` pair is a **reader**.
The rule: each reader reads a **disjoint** set of shards, and together they cover **all** of them.

Give each reader a global number and deal the shards like cards:

```
R         = world_size · num_workers           total readers
reader id = rank · num_workers + worker        0 … R − 1
reader r reads shards r, r + R, r + 2R, …
```

```python
list(range(12))[2::4]                      # reader 2 of 4: shards [2, 6, 10]
```

If every worker of a rank computes the same list, the same data is read several times and the
rest never. Nothing crashes; the model just sees a fraction of the data, repeated. That is
today's review bug.

### The shuffle buffer

Reading shards in order gives highly correlated consecutive samples. A stream cannot be
shuffled globally: it does not fit in memory. A **shuffle buffer** approximates it. Keep `size`
items; once the buffer is full, each incoming item takes the place of a randomly chosen buffered
item, which is emitted. At the end, the remaining items are emitted. A larger buffer mixes
better and uses more memory.

### Traps of the day

- Megabits read as megabytes.
- The number of cores rounded down.
- Readers that overlap, or shards nobody reads.
- A shuffle buffer that never emits its last items.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `raw_bytes_per_hour` | uncompressed size of one camera-hour | storage |
| `compressed_tb` | terabytes for hours × cameras at a bitrate | storage |
| `expected_decoded_frames` | average decode cost of a random frame | the cost of a random frame |
| `decode_cores_needed` | CPU cores for a sampling rate | the cost of a random frame |
| `assign_shards` | the shards of one `(rank, worker)` | sharding |
| `ShuffleBuffer` | streaming shuffle | the shuffle buffer |

Run: `uv run pytest weeks/w13-robot-data/d4-video-at-scale -k "not review"`

---

## Day 5: The data flywheel, a design

Folder: `d5-flywheel`. Exercise: `w13d5_flywheel.py`. Plus `design.md` and `rubric.md`.

### The one idea

Once a policy runs, its own rollouts are the most valuable data you can get: they show the
states the policy **actually reaches** (week 6, day 5). The flywheel turns them into training data.

### Interventions

When the policy drifts, an operator takes over and corrects. From that moment the human's
actions are a demonstration of **recovery**, in exactly the states where the policy needs one.
The log has a flag per step: 1 while the human controls the arm.

Extract each takeover as a segment: find every maximal run of ones `[s, e)`, and start a few
frames earlier (`context`) so the sample includes the situation that led to the takeover. Two
segments that overlap or touch after adding context are merged into one.

**Label with the executed action.** The log holds two actions per step: what the policy
proposed, and what was actually executed (the human's, during a takeover). The training label is
the **executed** one. Labelling corrections with the policy's own proposals teaches the policy
to repeat what it was about to do when the human had to step in. It is the DAgger bug of week 6,
in its real-world form, and today's review bug.

### Failure mining

Most autonomous episodes are never looked at. Rank them for human review by a priority:

```
priority = failure score + w · uncertainty
```

The failure score comes from a monitor (week 6, day 3); the uncertainty says where the monitor
is unsure, which is where a human label teaches the most. Review the top `k`. A deterministic
tie-break (the smaller id first) keeps the queue reproducible.

### What to collect next

The next 100 demonstrations should go where they help most. For each start condition you know
its failure rate `p`, measured on `n` evaluation episodes. Score each condition by its failure
rate plus the uncertainty of that rate:

```
score = p + sqrt( p (1 − p) / n )
```

A condition that fails often gets data; so does one that was barely evaluated.

Split the budget in proportion to the scores. Proportions give fractions and demonstrations come
in whole numbers. The **largest-remainder** method: give each condition the integer part of its
quota, then hand the leftover units one by one to the largest fractional parts. The total is
exactly the budget.

```python
quota = np.array([3.6, 2.7, 3.7])                    # budget 10
base = np.floor(quota).astype(int)                   # [3, 2, 3], total 8
order = np.argsort(-(quota - base), kind="stable")   # largest remainders first: indices 1 and 2 (0.7 each)
```

### The design round

`design.md`: the data platform for a fleet. Fill it, present it in 10 minutes, score yourself
with `rubric.md`. The whole week is material: QA at upload, curation and versioning, mixture and
sampling, storage and loading budgets, the flywheel.

### Traps of the day

- Corrections labelled with the policy's actions.
- Segments without context, or overlapping segments counted twice.
- A review queue sorted by failure score alone: the uncertain cases are never seen.
- An allocation that does not sum to the budget.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `intervention_segments` | merged `[start − context, end)` segments | interventions |
| `correction_dataset` | one sample per segment, labelled with executed actions | interventions |
| `review_queue` | top `k` ids by priority | failure mining |
| `allocate_collection` | budget split by largest remainder | what to collect next |

Run: `uv run pytest weeks/w13-robot-data/d5-flywheel -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one prompt from `drills/live`, timer on, AI off.
- **Tick `PROGRESS.md`** and write the retro.

You are done with week 13 when you can do these without notes:

- [ ] list five automatic checks on an upload and what each one catches
- [ ] explain masking and write the robust z-score
- [ ] explain why a random episode split leaks, and how to split by group
- [ ] say what a dataset fingerprint and a removal ledger are for
- [ ] compute mixture probabilities with a temperature, and the per-item weights
- [ ] design an ablation whose result is more than seed noise
- [ ] estimate storage and decode budgets from resolution, fps, bitrate and GOP
- [ ] explain how corrections become training data, with the right labels

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/14-robot-data-engineering.md`
- Your own pipeline: `reference/journal/book/10b-the-data-pipeline.md`
- Previous week: [`../w12-eval-harness/README.md`](../w12-eval-harness/README.md) · Next: [`../w14-edge-deployment/README.md`](../w14-edge-deployment/README.md)
