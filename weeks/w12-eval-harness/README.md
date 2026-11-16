# Week 12: the evaluation harness (your flagship engine)

Primer: `knowledge/13-evaluation-at-scale.md`. Builds on W6 (Wilson, McNemar, paired bootstrap, closed-loop eval).

W6 taught you to compare two policies on a toy. A real comparison of four policies on an SO-101 is
where most published robot results quietly go wrong: the session drifts (battery, lighting, the
operator learns), the operator knows which policy is running, 50 rollouts cannot see a 10-point gap,
"failure" means five different things, latency is measured on a laptop, and nobody can rerun the
exact checkpoint six months later.

This week you build the harness that fixes each of these. `flagship/` then turns it into the real
experiment: ACT vs SmolVLA vs VLA-JEPA vs your own head on your arm, 50+ rollouts each.

## Start here

This week does not have a full lesson page yet (weeks 1–9 do). Until it does, the knowledge you need is in two places:

1. **Before day 1 (30–45 min):** read the primer(s): `knowledge/13-evaluation-at-scale.md`. Builds on W6 (Wilson, McNemar, paired bootstrap, closed-loop eval). Read the "Mechanism" parts; keep the questions for the end of the week.
2. **Each day:** the day folder's `README.md` is the lesson (one page: formulas and the build list).

Each day folder holds the same files as in weeks 1–9:

| File | You |
|---|---|
| `predict.md` | write guesses before reading |
| `README.md` | read |
| `predict_run.py` | run, compare with your guesses |
| `w12dN_*.py` | **write the code** (the `# TODO`s) |
| `test_w12dN_*.py` | run, never edit |
| `review_w12dN.py` + its test | **find and fix** the planted bug |
| `questions.md` | answer out loud |

Any other `.py`, `.xml` or `.md` file in a day folder is provided: read it, do not edit it (design days also have a `design.md` to fill and a `rubric.md`).

One hour a day, commands from the repo root:

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess | `predict.md` |
| 10–18 | Read | the day's `README.md` |
| 18–20 | Check | `uv run python weeks/w12-eval-harness/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w12-eval-harness/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w12-eval-harness/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w12-eval-harness/<day>` |

## The days

| Day | Lab | Review bug |
|---|---|---|
| d1 | protocol: counterbalanced blocked schedule, blinding, record validation, drift | policies run one after the other |
| d2 | several policies: McNemar per pair, Holm, rank probabilities, minimum detectable difference | bootstrap breaks the pairing |
| d3 | failure taxonomy: rater agreement, per-cause rates with CIs, stage funnel | rates computed over failures only |
| d4 | latency on the target: percentiles, deadlines, sync vs async chunk execution | p99 from mean + std |
| d5 | reproducibility and the report: fingerprints, run manifests, results table + **design** | checkpoint hashed through `str()` |
