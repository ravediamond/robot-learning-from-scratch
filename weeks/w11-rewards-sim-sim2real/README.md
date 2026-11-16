# Week 11: Reward models, simulation, sim-to-real

**Everything you need for this week is on this page.** You need week 1 (training loop), week 8
day 1 (Jacobian, damped least squares, PD control from day 3) and week 10 (what a reward is for).

RL needs a reward and a place to fail cheaply. This week you build both: learned rewards, the
piece that lets RL run on a real arm, and a MuJoCo model of an SO-101-like arm you can control
and invert.

**This week has two labs.** Days 3–5 (system identification, domain randomisation, sim-to-real
analysis) are on the roadmap (`ROADMAP.md`) and not written yet. Until then,
`knowledge/12-simulation-sim2real.md` covers what to say about them in an interview.

## Start here

1. Read [The week in one table](#the-week-in-one-table).
2. Each day, open that day's folder and follow [What to do each day](#what-to-do-each-day).
3. The lesson for each day is the matching section of this page: [Day 1](#day-1-reward-models),
   [Day 2](#day-2-mujoco-basics).

### What is in a day folder

| File | What it is | You |
|---|---|---|
| `predict.md` | small questions, answered before you read anything | write guesses |
| `README.md` | one-page summary of the day | read |
| `predict_run.py` | prints the real answers | run, compare |
| `w11dN_*.py` | **the exercise**, with `# TODO` | **write the code** |
| `test_w11dN_*.py` | tests | run, never edit |
| `review_w11dN.py` + its test | code with one planted bug | **find and fix** |
| `questions.md` | interview questions | answer out loud |

Provided: `d1-reward-models/reward_data.py` (the data), `d2-mujoco-basics/so101_like.xml` (the arm).

### What to do each day

One hour, commands from the repo root.

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess, before reading | `predict.md` |
| 10–18 | Read the lesson | the day's section on this page |
| 18–20 | Check your guesses | `uv run python weeks/w11-rewards-sim-sim2real/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w11-rewards-sim-sim2real/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w11-rewards-sim-sim2real/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w11-rewards-sim-sim2real/<day>` |

## The week in one table

| Day | Folder | You learn | You build | The review bug is about |
|---|---|---|---|---|
| Mon | `d1-reward-models` | where rewards come from on a real robot | a success classifier with a precision-first threshold; a progress model learned from order | which frames may be compared |
| Tue | `d2-mujoco-basics` | MuJoCo's objects, two clock rates, PD control, IK | control and inverse kinematics of a simulated arm | a rounding |

---

## Day 1: Reward models

Folder: `d1-reward-models`. Exercise: `w11d1_rewards.py`. Provided: `reward_data.py`.

Code in this section assumes:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Why

In last week's toy world the reward was `−distance`, handed to you. On a real robot nobody hands
you a reward. There are three practical sources, all **learned**:

1. **A success classifier.** A binary model on the final image or state: did the task succeed?
   The reward is sparse (one signal at the end) and cheap to label.
2. **A progress model.** A score that rises along a good trajectory. It is learned from **order**
   alone: inside one successful episode, a later frame should score higher than an earlier one.
   No reward label is needed.
3. **A value function** (week 10, day 5): expected steps to success, trained on successes and failures.

### Rule 1: RL exploits false positives

A classifier is never perfect. For a supervised task that is a small loss of accuracy. For RL it
is a hole: the agent searches for whatever gives reward, and a state the classifier wrongly
calls "success" is a place to collect reward forever, without doing the task.

So the two kinds of error are not equal. A **false positive** (reward without success) corrupts
learning; a **false negative** (a success not rewarded) only slows it.

- **Precision**: among the states called success, the fraction that really are.
- **Recall**: among the real successes, the fraction that are called success.

Raising the decision threshold raises precision and lowers recall. For a reward model you choose
the threshold to reach a required **precision**, then accept the recall that comes with it.

```python
scores = torch.tensor([0.9, 0.8, 0.6, 0.3]); labels = torch.tensor([1, 1, 0, 1])
pred = scores >= 0.6
(pred & (labels == 1)).sum() / pred.sum()            # precision at threshold 0.6: 2 of 3
```

Among the thresholds that reach the precision, the **smallest** one has the highest recall. If
none reaches it, the honest answer is "no threshold": infinity, the classifier never fires.

The classifier is trained with binary cross-entropy on its raw output:
`F.binary_cross_entropy_with_logits(logit, target)`. The model outputs **logits**; a sigmoid
turns one into a probability.

### Rule 2: compare only what is comparable

A progress model learns from pairs of frames: "this one is later than that one". Frame 10 of
episode A and frame 40 of episode B are **not** ordered: B may have started much closer to the
goal. Pairs must come from **inside one episode**. Pairs across episodes teach the model
something false. That is today's review bug.

The rows of today's data are stored time-major: row index `= t · n_episodes + e`. From a row
index, `e = index % n_episodes` and `t = index // n_episodes`. To draw a pair in the same
episode: draw an episode and two different times, then build the two row indices.

### The pairwise loss

For a pair `(a, b)` with label `y = 1` if `b` is later than `a`:

```
loss = BCE_with_logits( score(b) − score(a), y )
```

Read `score(b) − score(a)` as the logit of "b is later". The loss pushes the later frame's
score above the earlier one's. This is the Bradley–Terry model, also used to rank players from
match results. Only **differences** of scores are trained, so the absolute level of the score is
arbitrary.

### Checking a progress model: Spearman correlation

The score should rise with time inside an episode; its exact values do not matter. The
**Spearman correlation** measures agreement in **order**: replace each value by its rank, then
take the ordinary (Pearson) correlation of the ranks. 1 means the same order, 0 no relation.

```python
x = torch.tensor([0.1, 0.9, 0.5])
x.argsort().argsort()                     # tensor([0, 2, 1]): the rank of each element
```

Pearson correlation of `a` and `b`: centre both, then `sum(a·b) / sqrt(sum(a²) · sum(b²))`.

### Traps of the day

- A threshold chosen for accuracy.
- Pairs across episodes.
- The pair label reversed: the model then learns to count down.
- A sigmoid applied before `binary_cross_entropy_with_logits`.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `train_success_classifier` | an MLP that outputs logits, binary cross-entropy | rule 1 |
| `threshold_for_precision` | the smallest threshold that reaches the precision | rule 1 |
| `progress_pairs` | index pairs inside one episode, with their label | rule 2 |
| `pairwise_loss` | Bradley–Terry | the pairwise loss |
| `train_progress_model` | an MLP score trained on pairs | the pairwise loss |
| `spearman` | rank correlation | checking a progress model |

Run: `uv run pytest weeks/w11-rewards-sim-sim2real/d1-reward-models -k "not review"`

---

## Day 2: MuJoCo basics

Folder: `d2-mujoco-basics`. Exercise: `w11d2_mujoco.py`. Provided: `so101_like.xml`.

Today uses **numpy**, not torch. `so101_like.xml` is a 5-joint arm with the SO-101's
proportions, written for this repo. It is not the official model; for real work use the official
SO-101 MJCF from `TheRobotStudio/SO-ARM100`.

### What a physics simulator is

A simulator holds a description of bodies, joints and motors, and advances time in small steps:
from positions, velocities and motor torques now, it computes them a moment later. MuJoCo is the
standard one for robot learning.

### The two objects

| Object | Holds | Examples |
|---|---|---|
| `MjModel` | what does not change | masses, joint axes and ranges, the timestep, actuator limits |
| `MjData` | the state, and everything computed from it | `qpos` (joint positions), `qvel` (velocities), `ctrl` (motor commands), `time` |

```python
import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <body>
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0"/>
      <site name="tip" pos="0.3 0 0"/>
    </body>
  </worldbody>
  <actuator><motor joint="j" ctrlrange="-1 1"/></actuator>
</mujoco>
"""                                                    # a one-joint arm, enough to try the API
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
model.opt.timestep, data.qpos.shape, model.nu          # 0.002 s, one joint, one actuator
mujoco.mj_forward(model, data)                         # compute positions for the current state; time does not advance
mujoco.mj_step(model, data)                            # advance by one timestep
data.time                                              # 0.002
```

- **`mj_step`** advances the simulation by `model.opt.timestep`.
- **`mj_forward`** computes everything that depends on the current state (positions of bodies,
  Jacobians, forces) **without** advancing time. After you write `qpos` by hand, call it before
  reading any position.

### Two rates

Physics needs small steps to be stable: 2 ms here, 500 Hz. Your policy runs at 30 Hz, the camera
rate of the SO-101. One control step holds the same target for several physics steps:

```
substeps = control period / physics timestep = (1/30) / 0.002 = 16.67
```

It is not a whole number. Truncating gives 16: each control step then lasts 32 ms of simulated
time against 33.3 ms on the real robot, and the simulated clock runs 4% fast. Rounding to the
nearest gives 17. Truncation is today's review bug. The better fix is to choose a timestep that
divides the control period exactly, and to assert it.

### PD control and gravity

The simulated arm has **torque** motors: you command a torque, not a position. A position servo
is built from the PD controller of week 8:

```
τ = kp · (q_desired − q) − kd · q̇
```

The P term pulls toward the target; the D term brakes in proportion to the velocity.

Under gravity a pure PD controller stops short of the target: it needs a non-zero error to
produce the torque that holds the arm up. The steady error is about `τ_gravity / kp`. MuJoCo
computes the torque that gravity and motion effects exert on each joint, `data.qfrc_bias`.
Adding it to the command cancels them: **gravity compensation**. The PD part then only has to
correct the error.

Inside a control step, the torque is **recomputed at every physics substep** from the current
`qpos` and `qvel`: the servo loop runs at the physics rate. Torques are clipped to the actuator
limits, `model.actuator_ctrlrange` (one `[low, high]` row per actuator), with `np.clip`.

### Sites, the Jacobian and inverse kinematics

A **site** is a named point attached to a body. The arm has one called `"ee"` at the
end-effector.

```python
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
data.site_xpos[sid].copy()                              # world position (3,); copy, it is a view into data
jacp = np.zeros((3, model.nv))                          # position Jacobian, filled in place
mujoco.mj_jacSite(model, data, jacp, None, sid)         # None: the rotation Jacobian is not wanted
```

`data.site_xpos[sid]` is a **view** into MuJoCo's memory: it changes at the next step. Copy it
if you keep it.

Inverse kinematics is week 8's damped least squares, with MuJoCo providing the Jacobian:

```
dq = Jᵀ ( J Jᵀ + λ² I )⁻¹ ( target − ee )            np.linalg.solve for the inverse
```

Two guards for a real arm: scale `dq` down when its largest entry exceeds a maximum step, so one
iteration never jumps far; and clip the new joint angles to the joint ranges, `model.jnt_range`.
Iterate until the position error is below a tolerance.

An IK step in this lab must **not modify** the `data` it is given: read from it, return the new
joint angles. The solver works on its own fresh `MjData` and sets the pose with the provided
`set_q` between steps.

### See it move (optional, needs a display)

```bash
uv run python -m mujoco.viewer --mjcf=weeks/w11-rewards-sim-sim2real/d2-mujoco-basics/so101_like.xml
```

### Traps of the day

- `int()` on the number of substeps.
- Reading a position after writing `qpos` without `mj_forward`.
- A view kept instead of a copy.
- The torque computed once per control step instead of once per substep.
- Unclipped torques: the simulated arm does what the real motors cannot.

### Today's build

| Function | Job | Lesson section |
|---|---|---|
| `n_substeps` | physics steps per control step, rounded, at least 1 | two rates |
| `pd_torque` | PD, plus the bias when given | PD control and gravity |
| `control_step` | one control period: torque, clip, step, at every substep | PD control and gravity |
| `ee_pos`, `ee_jacobian` | position and Jacobian of the `"ee"` site | sites, the Jacobian |
| `ik_step`, `solve_ik` | one damped step with the two guards; the iteration | inverse kinematics |

Run: `uv run pytest weeks/w11-rewards-sim-sim2real/d2-mujoco-basics -k "not review"`

---

## End of the week

You are done with week 11 when you can do these without notes:

- [ ] name three sources of reward on a real robot
- [ ] explain why a reward classifier is tuned for precision
- [ ] explain why progress pairs stay inside one episode, and write the pairwise loss
- [ ] say what `MjModel`, `MjData`, `mj_step` and `mj_forward` are
- [ ] compute the number of substeps and explain the clock drift
- [ ] explain the steady error of PD under gravity and how compensation removes it

## Optional depth (not needed for the labs)

- Interview questions: `knowledge/11-reinforcement-learning.md` (section 7), `knowledge/12-simulation-sim2real.md`
- Previous week: [`../w10-reinforcement-learning/README.md`](../w10-reinforcement-learning/README.md) · Next: [`../w12-eval-harness/README.md`](../w12-eval-harness/README.md)
