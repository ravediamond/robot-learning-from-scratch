# Week 9: Mock interviews

**Everything you need for this week is on this page.** This week adds almost nothing new: it
tests weeks 1–8 in the formats of a real interview loop. Each day says what the round is, how to
run it, what you must know cold, and where on the earlier week pages to re-read if it is not there.

## Start here

1. Read [The week in one table](#the-week-in-one-table) and [How this week is different](#how-this-week-is-different).
2. Each day, read that day's section here **first**, then open the folder.
3. Sections: [Day 1](#day-1-distributed-training-and-exact-resume), [Day 2](#day-2-timed-coding),
   [Day 3](#day-3-fix-the-script), [Day 4](#day-4-paper-to-experiment-to-verdict),
   [Day 5](#day-5-system-design-pushback-final-retro).
4. Friday: [End of the course](#end-of-the-course).

## The week in one table

| Day | Folder | The round | Format | Files you write |
|---|---|---|---|---|
| Mon | `d1-ddp-resume` | "what breaks going from 1 to 8 GPUs?" | a lab with tests, as in weeks 1–8 | `w9d1_ddp.py`, `review_w9d1.py` |
| Tue | `d2-timed-coding` | live coding | 45 min, four blank files, timer | `m1_attention.py` … `m4_bootstrap.py` |
| Wed | `d3-fix-the-script` | debugging | 45 min, one script with 7 bugs | `broken_bc.py`, `FIXES.md` |
| Thu | `d4-research-verdict` | research discussion | 30 min written + 20 min code | `paper_to_experiment.md`, `w9d4_verdict.py` |
| Fri | `d5-system-design-pushback` | system design, pushback, opener | three recorded rounds, no tests | `system_design.md`, `pushback.md`, `final_retro.md` |

## How this week is different

- **No `predict.md`** and, except Monday, no lesson to read before starting. The round is the
  cold attempt.
- **Timer on, AI fully off, no notes, no documentation, no autocomplete.** As in the room.
- **Talk out loud.** Shapes as you type, edge cases before you run, the test you are about to
  run and why. Record yourself at least once; listening to it is the fastest feedback you will get.
- **Over budget means redo tomorrow from a blank file.** Write the time taken on the first line.
- Start each day with the spaced retrieval as usual: `uv run python scripts/review.py`.

The sections below hold the revision material. On Tuesday to Friday, read them **after** the
round, to repair what failed, never before.

All code on this page assumes:

```python
import torch
import torch.nn as nn
```

---

## Day 1: Distributed training and exact resume

Folder: `d1-ddp-resume`. Exercise: `w9d1_ddp.py`. A normal lab: lesson, build, review, questions.

### Why this matters

"What breaks when you go from 1 GPU to 8?" is asked in almost every research-engineer loop.
Today you run distributed training for real, with two processes on your CPU, and prove it
equals the single-process run.

### Distributed data parallel (DDP) in five facts

1. **One process per device.** `world_size` processes, each with a number, its **rank**, a full
   copy of the model, and its own share of the data.
2. **Gradients are averaged across processes.** After `backward()`, each process sends its
   gradients and receives the average (an **all-reduce**). Every copy then takes the **same** step,
   so the copies stay identical.
3. **It equals one big batch.** With a mean loss per process and shares of equal size, two
   processes on 16 samples each compute exactly the gradient of one process on the 32. That
   equivalence is today's test.
4. **The data must be shared out.** Each rank takes a disjoint part of each batch. In this lab:
   rank `r` takes `idx[r::world_size]`, every `world_size`-th index starting at `r`. In real code
   a `DistributedSampler` does it, and needs `set_epoch(e)` each epoch or every epoch reuses the
   same shuffle.
5. **Only rank 0 logs and saves.** And every rank must call the collective operations in the
   same order, or the group waits forever.

The skeleton of a worker:

```
torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)     # gloo: the CPU backend
model = torch.nn.parallel.DistributedDataParallel(make_model(0))                   # wraps the model
… the ordinary loop: zero_grad, forward, loss, backward (the all-reduce happens here), step …
if rank == 0: torch.save(model.module.state_dict(), path)                          # .module: the unwrapped model
torch.distributed.destroy_process_group()
```

The processes find each other through two environment variables, `MASTER_ADDR` and
`MASTER_PORT`, set before `init_process_group`.

**`model.module`.** DDP wraps your model; the wrapper's `state_dict` has every key prefixed with
`module.`. Save `model.module.state_dict()`, or the checkpoint will not load into a plain model
(week 8, day 5).

**Worker functions must be defined at the top level of the file.** A new process receives its
function by name; a function defined inside another function cannot be sent.

### What breaks from 1 to 8 GPUs (the interview answer)

- **The effective batch is 8 times larger.** The learning rate that worked no longer fits;
  the usual rule is to scale it with the batch and lengthen the warm-up.
- **Data sharding.** Without it every process sees the same samples.
- **Epoch shuffling.** The forgotten `set_epoch`.
- **Logging and saving** from every rank: duplicated logs, corrupted files.
- **BatchNorm** statistics are per process unless converted to a synchronised version;
  LayerNorm and GroupNorm do not care.
- **Evaluation**: metrics computed on one rank's share, or shares of unequal size averaged as
  if equal.
- **Randomness**: the same seed on every rank gives every rank the same augmentations.

### Exact resume

A checkpoint that allows continuing **exactly** must hold everything that influences the next step:

| State | Why |
|---|---|
| model `state_dict` | the weights |
| optimiser `state_dict` | Adam's running averages and step count; without them the first steps after resume are different |
| scheduler `state_dict` | where the learning rate is on its curve |
| step counter | where to continue |
| random generator state | which batches come next: `gen.get_state()` / `gen.set_state(...)` |

```python
gen = torch.Generator().manual_seed(0)
state = gen.get_state()                       # a tensor: save it in the checkpoint
a = torch.randint(0, 100, (3,), generator=gen)
gen.set_state(state)                          # restore
torch.equal(a, torch.randint(0, 100, (3,), generator=gen))      # True: the same draws again
```

**The test of a resume**: train `2N` steps in one go; separately train `N`, save, throw
everything away, rebuild, load, train `N` more. The final weights must be **identical**, bit for
bit. If they are close and not equal, one piece of state is missing. The review file is missing
one; the table above tells you where to look.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `ddp_worker` | one rank: init, wrap, train on its share, rank 0 saves | DDP in five facts |
| `single_process_reference` | the same run in one process | fact 3 |
| `save_checkpoint`, `load_checkpoint` | everything needed to continue exactly | exact resume |
| `train_steps` | the loop between two step numbers | week 1 day 4, week 2 day 2 |

Run: `uv run pytest weeks/w9-mock-interviews/d1-ddp-resume -k "not review" -s`, then `-k review`.

---

## Day 2: Timed coding

Folder: `d2-timed-coding`. Four files with signatures only.

### The round

| File | Budget | You write | Test |
|---|---|---|---|
| `m1_attention.py` | 10 min | causal scaled dot-product attention; multi-head self-attention from raw weight matrices | `-k m1` |
| `m2_flow.py` | 12 min | flow-matching loss on action chunks; Euler sampler | `-k m2` |
| `m3_cem.py` | 13 min | CEM over action sequences with a dynamics function and a cost | `-k m3` |
| `m4_bootstrap.py` | 10 min | paired bootstrap interval of a success difference; Wilson | `-k m4` |

Run one: `uv run pytest weeks/w9-mock-interviews/d2-timed-coding -k m1`

Then `questions.md`: the follow-ups an interviewer asks once the code works.

### What is scored in the room

- You say every shape as you type it.
- You state the edge cases before running: empty input, a fully masked row, `k > n`, a
  resample with one class.
- You test on a tiny case you can check by hand, with a **different size on every axis** so a
  swapped axis is a shape error instead of a silent pass.
- You compare against a reference when one exists (`F.scaled_dot_product_attention`).
- You run the test only after saying what you expect.

### After the round: where each piece was taught

| Mock | Re-read |
|---|---|
| m1 attention, heads, causal mask | [week 3](../w3-transformers-gpt2/README.md), days 1–2 |
| m2 flow-matching loss and sampler, the `t = 0` noise convention, broadcasting `t` over `(B, H, D)` | [week 4](../w4-generative-flow/README.md), day 3; conditioning, day 4 |
| m3 CEM: sample, roll out, elites, refit, floor on the std | [week 7](../w7-jepa-world-models/README.md), day 5 |
| m4 paired bootstrap, Wilson | [week 6](../w6-robot-learning-eval/README.md), day 2 |

The ten timed prompts in `drills/live/` are more of the same, for the weeks after this course.

---

## Day 3: Fix the script

Folder: `d3-fix-the-script`. `broken_bc.py` trains a chunked behavior-cloning policy on the
obstacle world and saves a checkpoint. It runs end to end with no error. It contains **7 planted
bugs**, all of kinds you have met.

### The round

1. Timer, 45 minutes. **Read the whole script first.** Say out loud what each function should do.
2. For each bug: name it, fix it, write one line in `FIXES.md` (what it was, how it would show up).
3. `uv run pytest weeks/w9-mock-interviews/d3-fix-the-script` has one test per bug. The test
   names do not say which bug they target.
4. Count the bugs you found **before** running the tests. That count is your score.

### A method for reading a training script

Go through the script in the order the data flows, and at each stage ask the question that
stage usually gets wrong. Every item below was a lesson or a review bug in this course.

| Stage | Questions | Taught in |
|---|---|---|
| Seeds | set before the split? before the model is built? | week 2 day 1 |
| Split | by episode or by frame? validation ids disjoint from training ids? | week 1 day 5 |
| Windows | do chunks cross episode boundaries? is there a mask? | week 2 day 1 |
| Statistics | computed on the training split only? over real steps only? stored in buffers? | week 1 day 5, week 2 day 5 |
| Model | every layer registered? outputs reshaped, not permuted by accident? | week 1 days 1 and 4 |
| Loss | normalised units? masked, with the right denominator? `(N, 1)` against `(N,)`? | week 1 day 2 |
| Step | order zero → backward → clip → step → scheduler? scheduler in steps or epochs? | week 2 day 2 |
| Evaluation | `eval()` and `no_grad()`? augmentation off? mean weighted by batch size? back to `train()`? | week 1 day 4, week 2 day 5 |
| Deployment | actions un-normalised and clipped? | week 4 day 5 |
| Saving | statistics inside the checkpoint? strict load? outputs compared after reload? | week 1 day 5, week 8 day 5 |

In an interview, say the list out loud as you go. Finding four bugs with a visible method
scores higher than finding five by luck.

---

## Day 4: Paper to experiment to verdict

Folder: `d4-research-verdict`. Exercise: `w9d4_verdict.py`. Do it out loud, recorded.

### Part A (30 min, written): `paper_to_experiment.md`

Pick **one** claim from a paper you know (the file suggests four). Fill the template **before**
looking at any number. Its ten lines are the method of your own book, compressed:

| Line of the template | The question behind it |
|---|---|
| decision | what will you do differently if it passes, and if it fails? No decision, no experiment. |
| testable claim | the paper's sentence, rewritten as something measurable on **your** hardware |
| cheapest baseline | the simplest thing that could explain the result; it must not be a strawman |
| one factor | what changes; everything else is held fixed |
| metric, unit, pairing | success rate over episodes, on the same starts for both arms |
| `n` and why | the arithmetic of week 6 day 2, shown |
| pass line, fail line | two numbers chosen **before** the data, tied to the decision |
| verdict map | the confounds that would make any result "not proven" |
| instrument checks | strict load, parity, a sanity measurement: before reading the result |
| cost and stop rule | robot hours, GPU cost, when you stop |

Writing the pass and fail lines before the experiment is **pre-registration**. It is what stops
you from deciding afterwards that whatever you got was what you were looking for.

### Part B (20 min, code): the three verdicts

Your rule: an improvement `d` (method minus baseline) with its 95% interval `[lo, hi]`, a pass
line `P` and a separate, lower fail line `F`.

| Verdict | Condition |
|---|---|
| **pass** | `d ≥ P` and `lo > 0` |
| **fail** | `d ≤ F` and `hi < P` |
| **not proven** | everything else, and **always** when an unremovable confound is listed |

Two lines instead of one threshold, because "did not clearly pass" and "clearly failed" are
different findings. A small experiment mostly lands in "not proven", and that is an honest
result: it says the experiment could not tell.

`paired_diff_ci` is the paired bootstrap of week 6, day 2. Then run your rule on
`mock_experiment.py`: a "+15 points" claim tested on 30 paired starts. Predict the verdict first.

Run: `uv run pytest weeks/w9-mock-interviews/d4-research-verdict -s`

### After the round

Re-read: [week 6](../w6-robot-learning-eval/README.md), day 2 (sample sizes, pairing) and day 4
(what an evaluator must hold fixed).

---

## Day 5: System design, pushback, final retro

Folder: `d5-system-design-pushback`. No tests. Three recorded rounds and a retro. Timer and paper.

### Round 1: system design (35 min), `system_design.md`

"Design the data flywheel for a fleet of 10 low-cost arms doing a household tidying task at
customer sites." Fill the template, then present it in 10 minutes, recorded.

A **data flywheel** is the loop: robots run a policy → their episodes are logged → some become
training data → a new policy is trained → it is evaluated → it is deployed → the robots run it.
The template walks around that loop. What earns points:

- **Requirements first.** Task, definition of success, latency, compute per arm, connectivity,
  privacy. Ask before designing.
- **Numbers.** Bytes per arm per day, hours of human review per week, control rate, cost.
  A rough number beats an adjective.
- **Where the humans are**: corrections during operation, labelling, approval of a release.
- **A gate before redeploy**: fixed starts, paired comparison against the current policy, a bar
  (week 6 day 4).
- **Failure and safety**: monitors with their baseline (week 6 day 3), a canary arm, a rollback rule.
- **What you would build first**, and which result in the first two weeks would change the plan.

Score yourself 0–2 on each line of the rubric at the top of the file, after listening to the recording.

### Round 2: pushback (20 min), `pushback.md`

Five statements an interviewer makes to see whether you fold. 60 seconds each, out loud. Two
acceptable outcomes: you hold your position **with evidence** (a number of yours, with its
interval), or you update **with a reason**. Agreeing to be agreeable is the failing answer;
so is defending a number you cannot re-derive (week 8 day 4).

### Round 3: your opener (5 min)

The two-minute answer to "tell me about your work". The script and the drills are in
`reference/journal/book/exercises/interview/opener.md`. Record it twice, keep the better one.

### The retro: `final_retro.md`

Pick the three mocks or labs you did worst on (from `PROGRESS.md`), redo them, and write the retro.

---

## End of the course

- **AI-off gate:** today's rounds were the gate. Log them in `PROGRESS.md`.
- **Reading slot (20 min, read only):** `reading/README.md`, row 9 (what your harness guarantees and what it does not).
- **Keep the cards alive:** `uv run python scripts/review.py` daily for the next weeks. The 21- and 60-day boxes are where long-term retention is built.

You are ready for the loop when you can:

- [ ] write attention, a flow-matching step, CEM and a paired bootstrap from blank files inside their budgets
- [ ] find most of the planted bugs in a training script by reading, with a stated method
- [ ] turn a paper's claim into a pre-registered experiment with a pass line and a fail line
- [ ] answer "what breaks from 1 to 8 GPUs" and "how do you resume exactly"
- [ ] present a system design with numbers in 10 minutes
- [ ] quote your own results with intervals and hold them under pushback

## What comes next

Part 2, weeks 10–14 (`ROADMAP.md`): reinforcement learning, reward models and simulation, the
evaluation harness, robot data engineering, deployment. For a data, evaluation or deployment
role, `RE_TRACK.md` gives the order.

## Optional depth (not needed for the rounds)

- Mixed question bank with answers: `knowledge/08-question-bank.md`
- The interview chapter of your book: `reference/journal/book/16-the-interview.md`
- Previous week: [`../w8-robotics-register/README.md`](../w8-robotics-register/README.md) · Next: [`../w10-reinforcement-learning/README.md`](../w10-reinforcement-learning/README.md)
