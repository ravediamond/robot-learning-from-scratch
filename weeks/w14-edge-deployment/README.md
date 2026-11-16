# Week 14: Deploying policies on the robot's computer

**Everything you need for this week is on this page.** You need week 1 day 5 (buffers, raw in and
raw out), week 4 day 3 (flow matching and Euler sampling), week 8 day 3 (chunk switching) and
week 12 days 4–5 (latency budgets, fingerprints).

A policy that wins in a notebook still has to run at 30 Hz on a Jetson next to a camera, a servo
bus and a human. This week covers the path from checkpoint to arm: export with numerical parity,
quantisation you can defend, real-time chunking so the arm does not jerk between chunks, a safety
layer between the policy and the motors, and the rollout of a new model to a fleet without
betting the fleet on it.

## Start here

1. Read [The week in one table](#the-week-in-one-table).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-export-with-numerical-parity),
   [Day 2](#day-2-quantisation), [Day 3](#day-3-real-time-chunking),
   [Day 4](#day-4-the-safety-layer), [Day 5](#day-5-fleet-rollout-a-design).
4. Friday: [End of the week](#end-of-the-week).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day, with its "staff angle" | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w14dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w14dN_*.py` | tests | run, never edit |
| `review_w14dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided: `d1-export-parity/tensorrt_on_orin.md` (notes for the Orin, not runnable on a Mac),
`d3-real-time-chunking/toy_flow.py`. Friday adds `design.md` and `rubric.md`. Day 1 needs `onnx`
and `onnxruntime`, installed by `uv sync`.

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w14-edge-deployment/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w14-edge-deployment/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w14-edge-deployment/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w14-edge-deployment/<day>` |

## The week in one table

| Day | Folder | The step | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-export-parity` | from PyTorch to a runtime graph | a deployable wrapper, an ONNX export, a parity check | a shape fixed at export |
| Tue | `d2-quantization` | from 32-bit floats to 8-bit integers | int8 quantisation, calibration, fake quantisation | a gradient that is zero |
| Wed | `d3-real-time-chunking` | a smooth seam between chunks | guided flow sampling conditioned on the previous chunk | an alignment in time |
| Thu | `d4-safety-layer` | between the policy and the motors | limits, rate limits, a watchdog, an e-stop latch | which timestamp is trusted |
| Fri | `d5-fleet-rollout` | from one robot to many | artifact check, shadow mode, a canary test; a design | what is compared |

Code on this page assumes:

```python
import math
import torch
import torch.nn as nn
```

---

## Day 1: Export with numerical parity

Folder: `d1-export-parity`. Exercise: `w14d1_export.py`.

### The one idea

On the robot, the policy does not run as your Python code. A deployment **runtime** (ONNX
Runtime, TensorRT) runs a **graph**: a fixed list of operations and weights, saved to a file.
**Exporting** turns the PyTorch module into that graph. It is where silent bugs enter.

### Trap 1: normalisation left behind

The policy was trained on normalised states and outputs normalised actions. If the statistics
live in the Python data pipeline, the exported graph does not contain them, and on the robot the
model receives raw degrees and returns numbers that are not joint targets.

The fix is the design of week 1, day 5: **one module** that does normalise → model →
unnormalise, with the statistics as **buffers**. Buffers are part of the module, so they are
saved with it and written into the exported graph as constants. That module is what you export.

### Trap 2: shapes frozen at export

The exporter runs the module once on an example input and records what happened (**tracing**).
Every size it saw becomes a constant, including the batch size of the example, unless you declare
that axis **dynamic**.

```python
import os, tempfile
net = nn.Linear(4, 2).eval()
path = os.path.join(tempfile.mkdtemp(), "net.onnx")
torch.onnx.export(net, (torch.randn(1, 4),), path, input_names=["obs"], output_names=["act"],
                  dynamic_axes={"obs": {0: "batch"}, "act": {0: "batch"}}, dynamo=False)
```

Without `dynamic_axes` the graph accepts a batch of 1 and nothing else: fine for one arm, broken
for two cameras or a batched evaluator. That is today's review bug.

Call `.eval()` before exporting. Dropout traced in training mode is exported as dropout.

### Parity

Run the **same inputs** through the PyTorch module (eval mode, no gradient) and through the
runtime, and compare the outputs.

```python
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
x = torch.randn(5, 4)
out = sess.run(None, {"obs": x.numpy().astype(np.float32)})[0]          # a numpy array
float(np.abs(out - net(x).detach().numpy()).max())                      # about 1e-7
```

In 32-bit floats the largest absolute difference should be around `1e-6`. Anything larger is a
bug until proven otherwise: an operation the runtime implements differently, a different default,
a layer exported in training mode. Parity is checked after every export, not once.

TensorRT, the runtime of the Orin, compiles an ONNX graph for one specific GPU; the notes are in
`tensorrt_on_orin.md`. The method is the same: export, compile, check parity on the device.

### Traps of the day

- Statistics outside the exported module.
- Batch axis not declared dynamic.
- Export in training mode.
- Parity checked on one input, or with a tolerance chosen to make it pass.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `DeployablePolicy` | normalise → core model → unnormalise, statistics as buffers | trap 1 |
| `export_onnx` | eval mode, export, optional dynamic batch | trap 2 |
| `run_onnx` | run the graph with onnxruntime | parity |
| `parity` | largest absolute difference between PyTorch and the graph | parity |

Run: `uv run pytest weeks/w14-edge-deployment/d1-export-parity -k "not review"`

---

## Day 2: Quantisation

Folder: `d2-quantization`. Exercise: `w14d2_quant.py`.

### The one idea

A weight is stored as a 32-bit float. **Quantisation** stores it as an 8-bit integer: a quarter
of the memory, and integer arithmetic that edge hardware runs much faster. Each weight is
replaced by the nearest point of a grid:

```
scale = max|w| / 127
q     = clamp( round(w / scale), −127, 127 )         stored, int8
w ≈ scale · q                                        used
```

The grid has 255 points between `−max|w|` and `+max|w|`. The rounding error of one weight is at
most half a grid step, `scale / 2`. A large `scale` means a coarse grid.

```python
w = torch.tensor([0.10, -0.50, 1.27])
scale = w.abs().max() / 127                          # 0.01
q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)       # tensor([ 10, -50, 127])
q.float() * scale                                    # back to floats: the round trip
```

### Choice 1: one scale per tensor, or one per channel

A weight matrix has one row per output unit (channel). With **one scale for the whole matrix**,
a single row with large values sets the grid for all rows: the rows with small values are
rounded on a grid far too coarse for them, and some round to zero.

**Per-channel** quantisation gives each row its own scale: `max|w|` taken along every axis
except the first, with the result shaped `(C, 1, …)` so it broadcasts. A row of all zeros would
give a scale of 0 and a division by zero; give it a scale of 1.

### Choice 2: calibrating activations

Weights are known. **Activations** (the values flowing between layers) depend on the input, so
their range is measured on real data: **calibration**.

- **Min-max**: use the largest absolute value seen. One rare outlier sets the grid, and the
  ordinary values share a few grid points.
- **Percentile**: use the 99.9th percentile of the absolute values. The rare outliers are
  clipped; everything else keeps its resolution.

`torch.quantile(x.abs().flatten().float(), 0.999)`.

### Training with quantisation in the loop

**Post-training quantisation (PTQ)** quantises a trained model as it is. When that costs too
much accuracy, **quantisation-aware training (QAT)** lets the model adapt: during training the
forward pass quantises and immediately dequantises (**fake quantisation**), so the network sees
the rounding error and learns weights that survive it.

The obstacle: rounding has a gradient of zero almost everywhere. A step function is flat. With
the true gradient nothing upstream would learn. The **straight-through estimator** pretends, in
the backward pass only, that the rounding was the identity:

```python
x = torch.tensor([0.26], requires_grad=True)
xq = torch.round(x * 10) / 10            # a rounding: forward value 0.3
y = x + (xq - x).detach()                # forward: 0.3; backward: the gradient of x alone
y.sum().backward()
x.grad                                   # tensor([1.]): the gradient passed straight through
```

`(xq − x).detach()` is a constant: it shifts the value to the quantised one and contributes no
gradient. Returning `xq` directly gives a zero gradient and a model that silently stops learning
in every layer before it. That is today's review bug.

### Traps of the day

- One scale for a matrix with rows of very different sizes.
- Min-max calibration on data with outliers.
- No straight-through estimator.
- Accuracy "after quantisation" measured as a weight error only. Always re-run the closed-loop
  evaluation with the quantised model.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `quantize`, `dequantize` | symmetric int8, per tensor or per channel | the one idea, choice 1 |
| `percentile_scale` | a scale from a percentile of the absolute values | choice 2 |
| `fake_quant` | quantise–dequantise forward, identity gradient | training with quantisation |
| `quantize_linear_weights` | round-trip every Linear in place, report the relative error | choice 1 |

Run: `uv run pytest weeks/w14-edge-deployment/d2-quantization -k "not review"`

---

## Day 3: Real-time chunking

Folder: `d3-real-time-chunking`. Exercise: `w14d3_rtc.py`. Provided: `toy_flow.py`.

### The problem at the seam

Asynchronous inference (week 12, day 4) keeps the arm moving while the next chunk is generated.
Two problems remain where the new chunk takes over:

1. The new chunk was computed from an observation that is `d` steps old. Its first `d` actions
   are for moments already past; the robot executed the **old** chunk's actions meanwhile.
2. A flow-matching policy samples. The new chunk may pick the other mode (around the obstacle on
   the left where the old one went right). The arm jumps.

Week 8's shift handles the first and its blend smooths the second after the fact. **Real-time
chunking (RTC)** removes the cause: it generates the new chunk so that it **agrees** with what
the robot is already committed to.

### The idea: inpainting

During the `d` steps of inference, the robot will execute actions that are already known: the
rest of the previous chunk. So part of the new chunk is known in advance, and generating the
rest so that it fits is **inpainting**, like filling the missing part of an image so that it
matches the visible part.

**Aligning the known part.** The previous chunk was indexed from its own start. `executed`
actions of it are done. What remains, `prev_chunk[executed:]`, lines up with index 0 of the new
chunk. It is shorter than a chunk, so it is padded by repeating its last action. Using the
previous chunk without this shift conditions the new chunk on actions from the past: the seam
jumps exactly as if nothing had been done. That is today's review bug.

**A soft mask** says how strongly each position of the new chunk is tied to the known actions:

```
weight 1                         for the first `fixed` positions (they will be executed during inference: not negotiable)
a linear decay from 1 toward 0   over the rest of the overlap with the previous chunk
weight 0                         after the overlap: free
```

**Guided sampling.** Recall the Euler sampler of week 4: at each step the network gives a
velocity `v`. A flow model also gives, at any step, its current estimate of the final result:
for the straight-line path, `x1_hat = x + (1 − t) · v`. Guidance adds a pull of that estimate
toward the known actions, on the masked positions:

```
v_guided = v + guidance · mask · ( target − x1_hat )
```

Where the mask is 1 the estimate is pulled hard to the known actions; where it is 0 the sample
is free. Because all positions are generated **together**, the free part comes out coherent with
the fixed part: if the prefix goes left, the rest goes left.

(The published method uses a more principled correction; the masked pull with a fixed strength
shows the same effect.)

`toy_flow.py` provides an exact velocity field for a chunk distribution with two modes, up or
down, so nothing needs training. You measure the **seam jump**: the size of the step between the
last old action executed and the first new action used, with and without guidance.

### Traps of the day

- The previous chunk not shifted.
- A mask that is 1 over the whole chunk: the new chunk copies the old one and never uses the new observation.
- Guidance on the velocity toward the target instead of on the estimate `x1_hat`.
- The seam measured at index 0 instead of at the switch.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `shift_previous` | the remaining actions of the previous chunk, aligned and padded | aligning the known part |
| `soft_mask` | weights 1, then a decay, then 0 | a soft mask |
| `guided_sample` | Euler sampling with the guided velocity | guided sampling |
| `seam_jump` | the step at the switch | the problem at the seam |

Run: `uv run pytest weeks/w14-edge-deployment/d3-real-time-chunking -k "not review"`

---

## Day 4: The safety layer

Folder: `d4-safety-layer`. Exercise: `w14d4_safety.py`.

### The one idea

A learned policy will, sooner or later, output something wrong: a NaN after a bad frame, a jump
to the other side of the workspace, a command into a joint limit, or a perfectly good action
computed from a camera frame that is 800 ms old because the USB hub hiccupped.

The **safety layer** is small, boring, deterministic code between the policy and the motors.
It contains no learning, so it can be tested exhaustively. It is what lets you run an
experimental policy next to a person.

### The five guards

Today uses **numpy**: commands are small arrays, one value per joint.

**1. NaN and inf guard.** A non-finite command must never reach a motor. If any value is not
finite, send the last safe command again (**hold**). `np.isfinite(cmd).all()`.

**2. Joint limits.** Clamp each joint into its calibrated range, with a margin on each side:
`[low + margin, high − margin]`. `np.clip` accepts arrays as bounds; limits are a `(J, 2)` array.

**3. Rate limit.** No joint moves more than `max_step` in one control step. Move from the
previous command toward the new one by at most that much:

```python
import numpy as np
prev = np.array([0.0, 10.0]); cmd = np.array([5.0, 9.5]); max_step = 1.0
prev + np.clip(cmd - prev, -max_step, max_step)            # array([1. , 9.5])
```

A wild output then becomes a slow drift that the operator has time to stop.

**4. Watchdog on stale observations.** If the observation the action was computed from is older
than `max_age`, hold. The age is measured from the moment the camera **captured** the frame, not
from the moment the message arrived. A frame can sit in a buffer for a second and then arrive
looking fresh. Trusting the arrival time is today's review bug: the watchdog then never fires in
exactly the situation it exists for.

**5. E-stop latch.** Once the emergency stop is triggered, the layer stays stopped until an
explicit reset, even if the trigger goes away. A **latch** is state that stays set. Without it, a
flickering signal would stop and restart the arm repeatedly.

### Putting them in order

The filter is an object with state: the last safe command and the latch. On each call it
applies the guards in a fixed order and stops at the first one that says "hold":

```
latched               → hold the last safe command, record "latched"
non-finite command    → hold, record "nonfinite"
stale observation     → hold, record "stale"
otherwise             → clamp into the limits, then rate-limit from the last safe command
```

The checks that can only say "hold" come first. Then the command is made legal (limits), and
last, the arm moves toward that legal target by at most one step. The last safe command is itself
inside the limits, so a bounded step from it toward a point inside the limits stays inside.

Every hold is recorded in a list of **events**. The record matters as much as the command: "held
for staleness 40 times in this episode" is a finding about the camera pipeline, not about the policy.

The filter returns a **copy** of the command. Returning its own internal array would let the
caller modify the filter's memory of the last safe command by accident.

Because every guard is a small pure function, the tests can cover every branch. That is the
argument for keeping this layer free of anything learned.

### Traps of the day

- Staleness from the receipt time.
- An e-stop that releases by itself.
- A NaN compared with `==`: NaN is not equal to anything, itself included.
- "Hold" that returns zeros instead of the last safe command: the arm drops to its zero pose.
- The internal array returned without a copy.

### Today's build

| Piece | Job | Lesson section |
|---|---|---|
| `clamp_limits` | clamp into the calibrated range with a margin | guard 2 |
| `rate_limit` | at most `max_step` per joint per step | guard 3 |
| `is_stale` | capture time older than `max_age` | guard 4 |
| `SafetyFilter` | `estop`, `reset`, `step`: all five guards, with state and events | guard 5, putting them in order |

Run: `uv run pytest weeks/w14-edge-deployment/d4-safety-layer -k "not review"`

---

## Day 5: Fleet rollout, a design

Folder: `d5-fleet-rollout`. Exercise: `w14d5_rollout.py`. Plus `design.md` and `rubric.md`.

### The one idea

A new policy that won the evaluation can still lose in the field: other lighting, other objects,
a site nobody evaluated. Roll it out like a risky software release, in four stages.

### Stage 1: the artifact check

Before anything runs, the robot proves it loaded **exactly** the approved weights: the
fingerprint of what is in memory equals the approved fingerprint (week 12, day 5). It catches a
stale file, a partial download and a partial load (week 8, day 5) in one comparison.

### Stage 2: shadow mode

The new policy runs on every observation, and its actions are **not executed**. The deployed
policy keeps control. You compare what the two would have done, step by step. A step
**disagrees** if any joint differs by more than a tolerance.

The disagreement rate, and above all the list of steps where it happens, tell you which
situations to look at before the new policy touches a motor. It costs compute and no risk.

```python
a = np.array([[0.0, 1.0], [2.0, 3.0]]); b = np.array([[0.0, 1.5], [2.0, 3.0]])
(np.abs(a - b) > 0.1).any(axis=1)                # array([ True, False]): step 0 disagrees
```

### Stage 3: the canary

A few robots, the **canaries**, run the new policy. The signal is the number of human
interventions. 5 canaries against 45 other robots: comparing **counts** is meaningless, the
other group has nine times more robot-hours. Compare **rates**, interventions per hour.
Comparing counts is today's review bug.

Is a higher canary rate more than noise? Events that happen independently over time follow a
Poisson law, and that gives an exact test. If both groups had the **same** rate, each of the
`n = k_canary + k_base` interventions would fall in the canary group with probability equal to
its share of the hours:

```
p0 = h_canary / ( h_canary + h_base )
k_canary ~ Binomial(n, p0)
p-value = P( X ≥ k_canary ) = Σ_{i = k_canary}^{n} C(n, i) · p0^i · (1 − p0)^(n − i)        math.comb(n, i)
```

It is one-sided: you only ask whether the canary is **worse**. A small p-value means the canary
rate is higher than chance explains.

The decision rule is written **before** the rollout: wait until the canaries have enough hours;
roll back if the p-value is below `α`; otherwise continue. A rule decided afterwards is always
decided in favour of the new model.

### Stage 4: staged rollout

Increase the share of robots step by step (say 5%, 25%, 100%), applying the same rule at each
stage. The number of robots at a stage is `ceil(fraction · n_robots)`, at least 1, and never
fewer than at the previous stage.

### The design round

`design.md`: deploying a VLA on 20 robots. Fill it, present it in 10 minutes, score yourself with
`rubric.md`. This week and week 12 are the material: latency budget on the target, export and
parity, quantisation and its evaluation, chunking, the safety layer, the rollout and its rollback rule.

### Traps of the day

- Counts instead of rates.
- A two-sided test, or the wrong tail.
- A decision before the canaries have enough hours.
- A rollback rule invented after seeing the numbers.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `verify_artifact` | loaded fingerprint equals the approved one | stage 1 |
| `shadow_disagreement` | rate, step indices, largest difference | stage 2 |
| `rate_ratio_pvalue` | exact one-sided test that the canary rate is higher | stage 3 |
| `canary_decision` | wait, roll back, or continue | stage 3 |
| `stage_sizes` | robots at each stage | stage 4 |

Run: `uv run pytest weeks/w14-edge-deployment/d5-fleet-rollout -k "not review"`

---

## End of the week

- **AI-off gate (15 min):** one prompt from `drills/live`, timer on, AI off.
- **Tick `PROGRESS.md`** and write the retro.

You are done with week 14 when you can do these without notes:

- [ ] say what an export does, the two traps, and what parity means with a number
- [ ] write int8 quantisation; explain per-channel scales and percentile calibration
- [ ] write the straight-through estimator and say why it is needed
- [ ] explain real-time chunking as inpainting: shift, mask, guided sampling
- [ ] list the five guards of a safety layer and why the capture timestamp
- [ ] describe a rollout in four stages and the test behind a rollback

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/15-edge-deployment.md`
- Real-time execution papers in depth: `reference/journal/book/11b-chunking-and-edge-cost-papers.md`
- Previous week: [`../w13-robot-data/README.md`](../w13-robot-data/README.md) · Back to the list: [`../README.md`](../README.md)
