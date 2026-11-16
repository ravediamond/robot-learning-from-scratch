# Week 14: deploying policies on the robot's computer

Primer: `knowledge/15-edge-deployment.md`. Builds on W8 D3 (latency, chunk switching) and W12 D4 (latency on target).

A policy that wins in a notebook still has to run at 30 Hz on a Jetson next to a camera, a servo bus and
a human. This week covers the path from checkpoint to arm: export with numerical parity, quantisation
you can defend, real-time chunking so the arm does not jerk between chunks, a safety layer between the
policy and the motors, and the rollout of a new model to a fleet without betting the fleet on it.

## Start here

This week does not have a full lesson page yet (weeks 1–9 do). Until it does, the knowledge you need is in two places:

1. **Before day 1 (30–45 min):** read the primer(s): `knowledge/15-edge-deployment.md`. Builds on W8 D3 (latency, chunk switching) and W12 D4 (latency on target). Read the "Mechanism" parts; keep the questions for the end of the week.
2. **Each day:** the day folder's `README.md` is the lesson (one page: formulas and the build list).

Each day folder holds the same files as in weeks 1–9:

| File | You |
|---|---|
| `predict.md` | write guesses before reading |
| `README.md` | read |
| `predict_run.py` | run, compare with your guesses |
| `w14dN_*.py` | **write the code** (the `# TODO`s) |
| `test_w14dN_*.py` | run, never edit |
| `review_w14dN.py` + its test | **find and fix** the planted bug |
| `questions.md` | answer out loud |

Any other `.py`, `.xml` or `.md` file in a day folder is provided: read it, do not edit it (design days also have a `design.md` to fill and a `rubric.md`).

One hour a day, commands from the repo root:

| Min | Step | Command or file |
|---|---|---|
| 0–5 | Retrieve | `uv run python scripts/review.py` |
| 5–10 | Guess | `predict.md` |
| 10–18 | Read | the day's `README.md` |
| 18–20 | Check | `uv run python weeks/w14-edge-deployment/<day>/predict_run.py` |
| 20–45 | Build | `uv run pytest weeks/w14-edge-deployment/<day> -k "not review"` until green |
| 45–53 | Review | `uv run pytest weeks/w14-edge-deployment/<day> -k review` |
| 53–60 | Explain | `questions.md` out loud, then `uv run python scripts/review.py add weeks/w14-edge-deployment/<day>` |

## The days

| Day | Lab | Review bug |
|---|---|---|
| d1 | export: a deployable wrapper with normalisation inside, ONNX, parity, dynamic batch | batch axis frozen at 1 |
| d2 | quantisation: per-tensor vs per-channel int8, calibration, fake-quant with a straight-through estimator | no straight-through gradient |
| d3 | real-time chunking: shifting the previous chunk, soft masks, guided flow sampling | previous chunk not shifted |
| d4 | the safety layer: limits, rate limits, stale-observation watchdog, NaN guard, e-stop latch | staleness from receipt time |
| d5 | fleet rollout: artifact checks, shadow mode, canary with a rollback test + **design** | counts compared instead of rates |

ONNX labs need `onnx` and `onnxruntime` (in `pyproject.toml`). TensorRT on the Orin is in `d1/tensorrt_on_orin.md`
(not runnable on a Mac).
