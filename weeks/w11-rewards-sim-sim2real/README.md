# Week 11: Reward models, simulation, sim-to-real

Primers: `knowledge/11-reinforcement-learning.md` §7 (rewards), `knowledge/12-simulation-sim2real.md`.

RL needs a reward and a place to fail cheaply. This week you build both: learned rewards (the piece that
lets RL run on a real arm) and a MuJoCo model of an SO-101-like arm you can control, differentiate and invert.

## Start here

This week does not have a full lesson page yet (weeks 1–9 do). Until it does, the knowledge you need is in two places:

1. **Before day 1 (30–45 min):** read the primer(s): `knowledge/11-reinforcement-learning.md` §7 (rewards), `knowledge/12-simulation-sim2real.md`. Read the "Mechanism" parts; keep the questions for the end of the week.
2. **Each day:** the day folder's `README.md` is the lesson (one page: formulas and the build list).

Each day folder holds the same files as in weeks 1–9:

| File | You |
|---|---|
| `predict.md` | write guesses before reading |
| `README.md` | read |
| `predict_run.py` | run, compare with your guesses |
| `w11dN_*.py` | **write the code** (the `# TODO`s) |
| `test_w11dN_*.py` | run, never edit |
| `review_w11dN.py` + its test | **find and fix** the planted bug |
| `questions.md` | answer out loud |

Any other `.py`, `.xml` or `.md` file in a day folder is provided: read it, do not edit it (design days also have a `design.md` to fill and a `rubric.md`).

One hour a day, commands from the repo root:

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess | `predict.md` |
| 10–18 | Read | the day's `README.md` |
| 18–20 | Check | `uv run python weeks/w11-rewards-sim-sim2real/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w11-rewards-sim-sim2real/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w11-rewards-sim-sim2real/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w11-rewards-sim-sim2real/<day>` |

## The days

| Day | Lab | Review bug |
|---|---|---|
| d1 | success classifier, precision-first threshold, pairwise progress reward | pairs built across episodes |
| d2 | MuJoCo: PD control with substeps, gravity compensation, Jacobian, IK | wrong number of substeps |

Days 3–5 (system identification, domain randomisation, sim-to-real analysis) are on the roadmap
(`ROADMAP.md`); until then, primer 12 covers what to say about them.

MuJoCo labs use numpy, not torch. `so101_like.xml` is a 5-joint arm with the SO-101's proportions, written
for this repo: it is NOT the official model. For real work use the official SO-101 MJCF from
`TheRobotStudio/SO-ARM100`.
