# Week 13: robot data engineering

Primer: `knowledge/14-robot-data-engineering.md`. Builds on W2 D4 (reading LeRobot data, gaps, idle starts, lag).

In robot learning the data is the model's ceiling, and it is collected by tired humans on hardware
that drifts. Research teams now hire for this explicitly ("Research Engineer – Data"): someone who can
say what is in 10,000 hours of teleoperation, what should not be, what to collect next, and prove that
a mixture change helped. This week builds that toolkit.

## Start here

This week does not have a full lesson page yet (weeks 1–9 do). Until it does, the knowledge you need is in two places:

1. **Before day 1 (30–45 min):** read the primer(s): `knowledge/14-robot-data-engineering.md`. Builds on W2 D4 (reading LeRobot data, gaps, idle starts, lag). Read the "Mechanism" parts; keep the questions for the end of the week.
2. **Each day:** the day folder's `README.md` is the lesson (one page: formulas and the build list).

Each day folder holds the same files as in weeks 1–9:

| File | You |
|---|---|
| `predict.md` | write guesses before reading |
| `README.md` | read |
| `predict_run.py` | run, compare with your guesses |
| `w13dN_*.py` | **write the code** (the `# TODO`s) |
| `test_w13dN_*.py` | run, never edit |
| `review_w13dN.py` + its test | **find and fix** the planted bug |
| `questions.md` | answer out loud |

Any other `.py`, `.xml` or `.md` file in a day folder is provided: read it, do not edit it (design days also have a `design.md` to fill and a `rubric.md`).

One hour a day, commands from the repo root:

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess | `predict.md` |
| 10–18 | Read | the day's `README.md` |
| 18–20 | Check | `uv run python weeks/w13-robot-data/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w13-robot-data/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w13-robot-data/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w13-robot-data/<day>` |

## The days

| Day | Lab | Review bug |
|---|---|---|
| d1 | dataset QA at scale: frozen cameras, dead joints, limits, robust outliers, duplicates | outliers by mean ± 3σ |
| d2 | curation: near-duplicates, removal ledger, group-aware splits, dataset fingerprint and card | random episode split leaks sessions |
| d3 | mixtures and scaling: temperature sampling, per-item weights, scaling-law fit, ablations with seeds | source weight not divided by its size |
| d4 | video and storage at scale: budgets, keyframe-aware sampling, worker sharding, shuffle buffers | every worker reads the same shards |
| d5 | the flywheel: intervention segments, failure mining, what to collect next + **design** | corrections labelled with the policy's actions |
