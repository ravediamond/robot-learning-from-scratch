# Week 10: Reinforcement learning, from returns to RL on top of imitation

Primer: `knowledge/11-reinforcement-learning.md`.

Almost every robot-learning posting names RL next to imitation learning, and the 2025–2026 frontier
result (π*0.6 with RECAP) is "imitation first, then improve with RL from the robot's own experience".
This week you write the pieces by hand, small enough to run on a Mac in seconds.

All labs share one toy world, `rl_envs.py` (provided): `Reach2D`, a batch of 2-D points that must reach
a goal. obs = (pos, goal), action = velocity in [-1, 1]², reward = −distance, fixed horizon of 20 steps.

## Start here

This week does not have a full lesson page yet (weeks 1–9 do). Until it does, the knowledge you need is in two places:

1. **Before day 1 (30–45 min):** read the primer(s): `knowledge/11-reinforcement-learning.md`. Read the "Mechanism" parts; keep the questions for the end of the week.
2. **Each day:** the day folder's `README.md` is the lesson (one page: formulas and the build list).

Each day folder holds the same files as in weeks 1–9:

| File | You |
|---|---|
| `predict.md` | write guesses before reading |
| `README.md` | read |
| `predict_run.py` | run, compare with your guesses |
| `w10dN_*.py` | **write the code** (the `# TODO`s) |
| `test_w10dN_*.py` | run, never edit |
| `review_w10dN.py` + its test | **find and fix** the planted bug |
| `questions.md` | answer out loud |

Any other `.py`, `.xml` or `.md` file in a day folder is provided: read it, do not edit it (design days also have a `design.md` to fill and a `rubric.md`).

One hour a day, commands from the repo root:

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess | `predict.md` |
| 10–18 | Read | the day's `README.md` |
| 18–20 | Check | `uv run python weeks/w10-reinforcement-learning/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w10-reinforcement-learning/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w10-reinforcement-learning/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w10-reinforcement-learning/<day>` |

## The days

| Day | Lab | Review bug |
|---|---|---|
| d1 | returns, GAE, REINFORCE with a baseline | returns leak across the episode boundary |
| d2 | PPO: clipped objective, advantage normalisation, the update loop | `max` instead of `min` in the clip |
| d3 | SAC: tanh-Gaussian log-prob, twin Q target, soft update, temperature | missing tanh correction |
| d4 | offline RL: IQL (expectile value, advantage-weighted policy) beats BC on mixed data | expectile weights swapped |
| d5 | RECAP-style advantage conditioning on top of imitation | one global threshold for all tasks |
