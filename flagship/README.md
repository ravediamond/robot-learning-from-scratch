# Flagship: a fair, reproducible comparison of four policies on a real SO-101

The portfolio piece this repo builds towards. One artifact that shows the three things a robot
research team hires an engineer for: **reproducible** (every number traces to code, data, weights and
settings), **measurable** (success with uncertainty, failure causes, what the experiment can and
cannot detect), **deployable** (latency and execution behaviour measured on the robot's own computer).

## The experiment

- **Task**: pick the cube, place it in the bowl (the task of your `so101_bowls` data), SO-101 follower,
  policy running on the Jetson Orin.
- **Policies** (4): ACT, SmolVLA, VLA-JEPA, and your own head. Same training data for all four,
  same camera setup, each with its documented execution settings.
- **Rollouts**: 10 marked start configurations × 5 repeats = **50 per policy, 200 in total**,
  blocked and counterbalanced, operator blind to the policy (`PROTOCOL.md`).
- **Outputs**: success with Wilson CIs, pairwise McNemar with Holm, rank probabilities, failure causes
  as rates with CIs, stage funnel, latency p50/p95/p99 and deadline misses on the Orin, the minimum
  detectable difference stated up front, one manifest per session. Plus videos of typical failures.

## How the repo gets you there

| Step | What | Where |
|---|---|---|
| 0 | Build the harness by hand (it is the analysis engine below) | `weeks/w12-eval-harness` d1–d5 |
| 1 | Write and commit the pre-registration BEFORE any rollout | `PREREG.md` |
| 2 | Generate the blinded run sheet | `python flagship/make_sheet.py` |
| 3 | Dry-run the analysis on synthetic data | `python flagship/demo_records.py && python flagship/analyze.py --records flagship/demo/records.csv --aliases flagship/demo/aliases.json --latency flagship/demo/latency.csv --out flagship/demo/report.md` |
| 4 | Pilot: 2 policies × 20 rollouts, fix the protocol, re-commit it | `RUNBOOK.md` |
| 5 | Full sessions: 200 rollouts, manifests, latency probes | `RUNBOOK.md` |
| 6 | Report: generated tables + videos + protocol + prereg deviations | `analyze.py` → your project page |

`make_sheet.py`, `analyze.py` and `demo_records.py` import YOUR week-12 code: they work once those labs
are green (on `master`), or immediately on the `solutions` branch.

## Milestones

| Milestone | Done when |
|---|---|
| M0 harness | `uv run pytest weeks/w12-eval-harness` green, demo report generated |
| M1 pilot | 2 policies × 20 rollouts, protocol v2 committed with what the pilot changed |
| M2 full run | 200 rollouts, every session with a manifest, validation errors = 0 |
| M3 page | report + videos + repo link published; numbers in the page generated, not typed |

A pilot with two policies is already a credible portfolio item if it is honest about its n and its
minimum detectable difference. Do not wait for M3 to show it.
