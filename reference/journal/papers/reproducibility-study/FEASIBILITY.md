# Feasibility — "What survives when a paper leaves the lab": a pre-registered re-test of robot-learning papers on a ≈ €1,350 rig

WS-069, 2026-10-31. Memo only: nothing was run. Sources: the site register (`~/workspace/site/data/register.yaml`,
22 papers, 45 claim rows), `journal/EXPERIMENTS.md` (115 dated gate rows), `journal/METHOD.md`, the public repo
`paper-vs-hardware` (29 folders), WS-067 (reproducibility of the posts), WS-068 (audit of the paper-side fields), the
papers mirror (`papers-sync` 2026-10-31, 130 files).

## 1. Claim in one sentence

Of N robot-learning papers whose central claim was re-tested on an SO-101 arm and a Jetson Orin NX with a bar fixed
before the run, a minority reproduce (today 4 of 22 papers, 17 of 42 tested claim rows), and the failures fall into
three classes that the papers' own setups cannot reveal: edge latency, zero-shot judges, and latent planners beaten
by the commanded joints. True if the paper-level count holds at ≥ 20 papers with every bar's commit predating its
result, and the class assignment is reproducible by a second reader.

## 2. Evidence in hand

| what | n | result | anchor |
|---|---|---|---|
| papers in the register | 22 | 4 reproduce, 13 fail, 3 not proven, 1 not tested (D-JEPA), 1 withdrawn (TurboVLA) | `/papers/`, `data/register.yaml` |
| claim rows | 45 (42 tested; 37 paper claims, 8 ours) | 17 reproduce, 20 fail, 3 not proven, 4 not tested, 1 withdrawn | `public/data/claims.json` |
| their rig class | 22 | 9 sim, 7 > $20k, 3 $5–20k, 2 < $5k, 1 unstated | register `their_rig_cost_class` |
| contradiction pairs (paper vs rig, same triple) | 5 | WS-068: 1 like-for-like (ContactGuard AUROC), 3 against a baseline the paper never ran, 1 unchecked (PDF missing) | `journal/drafts/2026-10-31-claims-audit/REPORT.md` |
| L0 (every bar recomputed from published rows) | 29 posts | 29 / 29 PASS | WS-067 `L0.md` |
| L1 (gates rerun cold from the public repo) | 6 runnable | 6 / 6 reproduce (5 byte-identical); 12 blocked by an unpublished W1 environment; 11 need MAXN, a large GPU, a private checkpoint or the arm | WS-067 `REPORT.md` |
| paper-side fields checked against the PDFs | 12 of 22 papers | 46 OK, 20 citation fixes, 12 content rows to correct (3 number, 2 wording, 3 conditions, 4 not in the paper) | WS-068 `AUDIT.md` |
| compute | 22 papers | $45.10 of GPU and API in total, retries included | register `cost_usd` |

**Pre-registration.** 8 studies have a `PROTOCOL.md` committed before any number (AAC, horizon × depth, HWM, LeJEPA,
PACT, REMORA, smooth chunking, SURVEYOR; 2026-10-27 onward). The 14 papers first written up on 2026-10-23 rest on gates
run 2026-09-29 → 10-22 whose bars are in `EXPERIMENTS.md` rows; whether each bar's commit predates its result is
**unchecked** ($0: `git log -S` per row, ~2 h). **Blind labelling:** owner labels at rest for the arm gates; not blind to
arm in the earlier ones (to check per row). **Statistic:** the episode bootstrap (2,000 resamples, METHOD I.4) is used
by the 8 frozen protocols; the earlier gates mix bootstraps and point estimates, recomputable from stored rows ($0;
`shared/gate/data/*/` and the `expected/` folders of the 29 posts).

## 3. Gaps, costed

| gap | resource | queued? |
|---|---|---|
| bar-before-result check for the 14 early papers (commit order per gate) | ~2 h, $0 | new |
| recompute the early gates with the episode bootstrap from stored rows | ~1 Orin day, $0 | new |
| second reader for the failure classes (inter-rater agreement on 20 claim rows) | ~3 owner hours + a second person | new |
| finish the paper-side audit: 10 papers missing from the mirror | owner adds 11 PDFs, ~3 h reading | WS-068 follow-up |
| fix the 12 content rows of WS-068 | owner decisions, ~1 h | WS-068 follow-up |
| publish the W1 environment so 12 posts rerun at L1 | ~0.5 day + a 28 GB ghcr push | WS-067 follow-up |
| MAXN halves of 4 posts | ~2 Orin hours at MAXN, owner sudo | WS-067 follow-up |
| a second task family (today: bowls, cubes) for the arm gates | ~2 arm days + teleop | new |
| power: a paired 40-trial gate instead of 20 where a verdict hinges on it | 2–4 arm days | new |
| **total** | **≈ 5 arm days, ~3 Orin days, ~15 owner hours, $0–10** | |

## 4. Related work

| paper | what it showed | what differs here |
|---|---|---|
| 2606.08881 Benchmarking VLAs on SO-101: failure and recovery | a standardized real-world benchmark and failure taxonomy on the SO-101 | benchmarks policies; we re-test papers' claims against pre-set bars |
| 2607.24481 ArmnetBench v0.1 | 2,518 rollouts on a fleet of SO-101 cells, 12 tasks | fleet-scale evaluation of policies; we test the claims behind methods (judges, world models, latency) |
| 2603.13966 vla-eval | a harness decoupling inference from 14 simulation benchmarks | simulation only; no claim-by-claim verdicts |
| 2610.18869 AXIS-Bench (quality-aware VLA pretraining) | demonstration quality matters for pretraining | an empirical study within one method family |
| RoboSeed 2610.40026 | scores models on capability, affordability and research evidence | **not read here** (not in the mirror) — the closest statement of the thesis |
| Henderson et al. 2018, *Deep RL that matters* | seeds and implementation details swing RL results | not read here; same lesson on hardware rather than seeds |
| Pineau et al. 2021, *Improving reproducibility in ML research* | the NeurIPS reproducibility programme and checklist | not read here; we add the hardware gap and pre-registered bars |
| Agarwal et al. 2021, *Deep RL at the edge of the statistical precipice* | few-run evaluations need interval estimates | not read here; our episode bootstrap is the same remedy |
| ReScience C (Rougier et al. 2017) | a venue for replications | not read here; a possible venue model |

(The prompt also listed 2610.26625 EgoXperience and 2609.16705 The Robot Data Factory: in the mirror, but neither is a
reproducibility or benchmark-critique paper; not positioned against.)

## 5. Objections

**First: one rig, one task family, operator labels, n = 20 — this generalizes nowhere.** Answer: the claim is not
"these methods fail", it is "these claims did not survive *this* transfer, under a bar fixed before the run". The paper
states the condition line per verdict (their rig vs ours, 16 of 22 shown on sim or > $5k rigs) and reports the ones
that reproduced. Generality is the weakness to state, not to hide; the second task family in §3 narrows it.

**Second: three of the five headline contradictions oppose a baseline the papers never ran.** Answer (WS-068): true;
those are rig findings ("the commanded joints beat the world model here"), not refutations of a measured claim, and
the paper must say so. Only ContactGuard's is a like-for-like contradiction. Also: whether a 20-trial non-inferiority
gate can "fail" a paper — the paper reports power per gate and calls under-powered outcomes "not proven"; and the
author re-testing his own posts is not independent — WS-067's cold rerun from the public repo is the partial answer.

## 6. Venue

Primary: **TMLR** (rolling submissions, OpenReview, double-blind; has a Reproducibility certification; no hard page
limit). Fallback: a 2027 robotics workshop on evaluation or reproducibility at RSS or CoRL (deadlines to confirm when
announced). arXiv first: yes, after the §3 commit-order check (a pre-registration claim must be verifiable on day one).
Endorsement (cs.RO, first submission): **no contacts list exists in the repo; no endorser is named here.**

## 7. Minimal vs full paper

4 pages: the register table (22 papers, verdict, rig gap), the three failure classes with one example each, the
method (bar before run, cheapest baseline first), L0 / L1 reproducibility, limitations. Needs from §3: the commit-order
check. 8 pages: + per-class analysis, the contradiction pairs with WS-068's like-for-like distinction, the second task
family, inter-rater agreement, the episode-bootstrap recomputation.

## 8. Risk of a null result

The "claim" is a count; it cannot fail to exist. The risk is that the commit-order check shows bars set after results
for some early gates: those rows move to "exploratory", the count shrinks, and the paper reports both counts. A paper
with 8 strictly pre-registered studies is still publishable as a smaller study.

## 9. Recommendation

**go-after-X**, X = the commit-order check and the WS-068 content fixes (≈ 3 owner hours, $0). Then the 4-page version
can be written in ~1 week; the 8-page version needs the second task family (≈ 2 arm days at ~2 arm days per week).
Earliest realistic submission: arXiv + TMLR mid-December 2026 for the 4-page core, the 8-page version late January 2027.

## 10. Skeleton

Titles: (1) *What survives when a paper leaves the lab: re-testing 22 robot-learning papers on a ≈ €1,350 arm*;
(2) *Pre-registered re-tests of robot-learning claims on commodity hardware*; (3) *The cheapest baseline first: a
reproducibility study on an SO-101 and a Jetson*.

Abstract (≤ 150 words): Robot-learning papers are shown on Franka-class arms, data-centre GPUs or in simulation. We
re-tested the central claims of 22 papers on an SO-101 arm and a Jetson Orin NX (≈ €1,350), with the pass mark
committed before each run and the cheapest baseline run first. 4 papers reproduce, 13 fail, 3 are not proven
([to run]: commit-order check). The failures fall into three classes: models that do not meet a real-time budget on
the edge GPU, zero-shot judges beaten by episode length, and latent planners beaten by the commanded joints. Every bar
is recomputable from published rows (29 / 29 posts), and every runnable gate reproduces from the public repository
(6 / 6, [to run]: W1 environment for 12 more). Total compute: $45.

Sections: 1 Introduction (the transfer gap) · 2 Method (bar first, baseline first, register schema) · 3 The 22
papers (table) · 4 Three failure classes · 5 What the method itself revealed (contradiction pairs, the audit) · 6
Reproducibility of this study (L0 / L1) · 7 Limitations · 8 Conclusion.

Figures: (1) register table — `public/data/register.json` (today); (2) their rig cost class vs verdict — register
(today); (3) per-class examples: AAC 4,751 ms vs 298 ms budget (`expected/` of adaptive-chunking), Robometer vs
episode length 0.818 vs 0.938 (robometer post), G1b 0.60 vs 0.68–0.90 (leworldmodel post) — today; (4) the L1
reproducibility table — WS-067 `L1.md` (today; 12 rows `[to run]`).
