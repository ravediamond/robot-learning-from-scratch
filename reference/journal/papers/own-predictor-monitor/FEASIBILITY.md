# Feasibility — "The policy's own predictor as its failure monitor": zero extra parameters

WS-074, 2026-10-31. Memo only: nothing was run. Sources: `journal/NEXT_ANGLES.md` §29 (bar frozen 2026-10-25, SaiVLA-JEPA
note of 2026-10-31), `journal/EXPERIMENTS.md` (§29 rows, W1-A, W1-C2, REAP v1 / v2), `shared/gate/data/2026-10-24_vla_jepa_surprise/`,
the posts `vla-jepa` and `vla-jepa-surprise-robot-day`, the papers mirror.

**State of the evidence the paper depends on.** §29 step 2 — the robot day — **has run** (2026-10-27, WS-018): 20 of 20
rollouts succeeded, 9 of them on off-nominal layouts; by the frozen bar's own rule the result is **NOT PROVEN, too
few failures** (0 < 6). The paper's central experiment therefore exists and is empty; the question is how to get
failures, not whether to run.

## 1. Claim in one sentence

The latent-prediction error of a JEPA-trained VLA's own predictor, computed on the rollout's frames, catches ≥ 6 of 10
manipulation failures before their onset at ≤ 1/20 false interventions, with AUROC CI above 0.60 and ≥ 2 more
failures than the best baseline (W1 surprise, the logistic risk head, elapsed time, a joint-velocity stall rule) — with
zero extra parameters and no labels; paired with SaiVLA-JEPA's control-side null (+0.03 points), that makes the
predictor useless to act and useful to watch.

## 2. Evidence in hand

| step / gate | date | n | result | status | anchor |
|---|---|---|---|---|---|
| §29 step 0: feasibility | 10-25 | checkpoint `c65c98b` | predictor 157.45 M (F32), V-JEPA 2 ViT-L target encoder bundled (326 M), 739/739 world-model tensors load | done | `2026-10-24_vla_jepa_surprise/README.md` |
| §29 step 1: latency + exploratory replay | 10-25 | 50 checks; 10 trials (1 failure) | **1,973 ms** p50 per check at 25 W, 5.9 GB → offline only; failure max 1.546 vs successes 1.505–1.555 (no separation, n = 1) | measured, no claim | same |
| **§29 step 2: robot day** | 10-27 | 20 rollouts (9 off-nominal) | **20 / 20 successes, 0 failures** | **not proven: too few failures** | post `vla-jepa-surprise-robot-day`; folder `2026-10-29-vla-jepa-surprise-robot-day` |
| foil: a separate world model as screener (W1-A) | 10-19 | 50 (8 failures) | W1 L2 surprise 2/8 at 2/42 (AUROC 0.86); drift 0/8 | fail | `2026-10-19_w1_screening` |
| foil: FAIL-Detect / SAFE band on W1 (W1-C2) | 10-19 | 22/22 × 20 seeds | 46 % failures at 22.7 % successes flagged | fail | EXPERIMENTS |
| baseline to beat: REAP v2 risk head | 10-23 | 50 | 3/8 at 5/42 (AUROC 0.789); MLP 6/8 at 13/42 | fail | `2026-10-22_reap` |
| VLA-JEPA on the arm | 10-08 | 10 trials | 9/10 (the one failure is the replay's) | measured | post `vla-jepa` |
| leak fix in LeRobot (#4381, future-frame leakage) | 10-26 | 1 checkpoint | an upstream correctness fix made the checkpoint 47 % worse on held-out windows | published | post `upstream-fix-checkpoint-47-percent-worse` |

Pre-registered: §29's bar (frozen 2026-10-25, owner-approved; sealed 60-layout list, hash committed before rollout 1).
Blind: onsets owner-marked. Statistic: 5-fold by episode × 5 seeds, AUROC with episode bootstrap (as REAP).

## 3. Gaps, costed

| gap | resource | queued? |
|---|---|---|
| **≥ 6 failures** under the same bar: the bowls task is at ceiling for VLA-JEPA (20/20). Options: (a) the remaining 40 sealed layouts (the bar allows 60 rollouts) — likely still too few; (b) a harder task the checkpoint was not tuned on (cube among two: the 21-task SmolVLA got 2/6) — needs a VLA-JEPA fine-tune; (c) an earlier, weaker checkpoint. Any option other than (a) is a bar change and needs a new frozen protocol | (a) 1 arm day, $0; (b) ~$10 fine-tune on vast + 2 arm days; (c) $0 + 2 arm days | (a) within §29; (b), (c) new |
| the predictor on a 2nd policy checkpoint (does the signal survive the #4381 leak fix?) | ~0.5 Orin day | new |
| the "novel but fine" control: off-nominal successes must not trigger (the 9 off-nominal successes of 10-27 are this control's first data) | $0, offline on the stored frames | new |
| an online path (2 s per check at 25 W makes it post-hoc): MAXN measurement, fewer frames, or a smaller target encoder | ~1 Orin day | new |
| FARM comparison: our zero-parameter error vs FARM's 34k-parameter supervised readout on the same frozen states | ~1 Orin day (FARM's code is public) | new |
| **total** | **≈ 3–5 arm days, ~3 Orin days, $0–10** | |

## 4. Related work

| paper | what it showed | what differs here |
|---|---|---|
| 2609.11445 **FARM** | a 33,985-parameter supervised readout on frozen VLA-JEPA predictive states detects failures (pooled AUROC 85.68 OOF), incl. an SO-101 population; 0.23 ms per check | **the closest prior work**: same frozen states, but supervised; ours is the unsupervised prediction error, zero parameters, no labels. Whether FARM's 15 baselines include the predictor's own error is to check in its Table III |
| 2610.38508 SaiVLA-JEPA | a training-only future-latent head adds +0.03 points to control (p = 0.958) | the control-side null this paper pairs with |
| 2602.10098 VLA-JEPA | the policy, trained with a leakage-free latent world-model loss | the predictor is switched off at inference; we switch it back on |
| 2603.14482 V-JEPA 2.1 | dense predictive video features | the target encoder family |
| 2603.19312 LeWorldModel, 2607.25337 Temporal-Distance JEPA | small JEPA world models; plan-aware latent cost | the separate-model foil (W1) |
| 2608.13438 ContactGuard | probe on an imagined latent predicts grasp failure (0.98–0.99 AUROC, own dual-arm data) | a dedicated, trained monitor; here it reproduced in the opposite direction (current latent > imagined) |
| 2610.10515 RoboJEPA | scaling of action-conditioned latent predictors | a separate model |
| VLA-Corrector 2609.06508, SAFE 2506.09937, FAIL-Detect 2503.08558, planning limits of JEPA 2609.39235 | — | **not read here** (not in the mirror) |

(The prompt lists 2609.11445 as ReAP; in the mirror it is FARM, which matters more here than ReAP would.)

## 5. Objections

**First: one checkpoint, one task, ≤ 10 failures — and so far 0.** Answer: the bar's NOT PROVEN branch is exactly
this, and the paper cannot be written until failures exist; the gap in §3 is the whole paper. **Second: surprise may
be out-of-distribution detection, not failure detection, and a 2-layer head (or FARM's readout) beats it.** Answer:
the off-nominal successes are the control (novel but fine must not trigger); the claim is not "beats a trained head"
but "≥ 2 more catches than the baselines at zero parameters" — if FARM or REAP's head wins, the paper reports that.
Also: 2 s per check makes it a post-hoc monitor on the Orin; the title must say so unless the online path in §3
lands; and the #4381 leak fix is a confound for any checkpoint trained before it.

## 6. Venue

Primary (if PASS): a 2027 workshop on world models or robot failure detection (RSS / CoRL; to confirm). Fallback:
arXiv note paired with the WS-070 paper. Endorsement: none named (no contacts list in the repo).

## 7. Minimal vs full paper

4 pages (needs PASS): the bar, the 60-rollout day, caught / missed with onsets, the baselines, the off-nominal
control, SaiVLA-JEPA as the pairing. 8 pages: + FARM on the same states, a second checkpoint, the online path.

## 8. What the paper is under each outcome

- **PASS:** "The policy's own predictor screens its failures, better than a separate world model" — a decoupling
  result with SaiVLA-JEPA; workshop or short paper.
- **FAIL:** a negative result in the WS-070 paper (a section, not a paper): the predictor that does not help act also
  does not help watch, on this arm.
- **NOT PROVEN (today):** no paper. The robot-day post already published says what there is to say.

## 9. Recommendation

**no-go for now; go-after-X**, X = enough failures under a frozen bar: run the remaining 40 sealed layouts (1 arm day,
$0) and, if still < 6 failures, write a new protocol with a harder task or checkpoint (option (b) or (c), ~$10, 2 arm
days). Only then decide. Earliest submission if X yields a PASS: a 2027 workshop, February–spring 2027.

## 10. Skeleton

Titles: (1) *Useless to act, useful to watch: a VLA's own latent predictor as its failure monitor*; (2) *Zero
parameters, no labels: monitoring a JEPA-trained VLA with its own predictor*; (3) *The predictor the policy switched off*.

Abstract (≤ 150 words): JEPA-trained VLAs learn a latent predictor that is switched off at inference, and a matched
study found it adds nothing to control (+0.03 points). We switch it back on as a monitor: the predictor's error on the
rollout's frames, with no extra parameters and no labels. On an SO-101 with VLA-JEPA, [to run] of [to run] failures
are caught before onset at ≤ 1/20 false interventions (AUROC [to run]), against a separate world model (2/8 in a
previous study), a trained risk head (3/8) and elapsed time (0/8); off-nominal successes [to run]. The check takes
2 s on a Jetson Orin NX, so the monitor is post-hoc there.

Sections: 1 Intro (the decoupling) · 2 The predictor and its error · 3 Protocol (frozen bar, sealed layouts) · 4
Results `[to run]` · 5 Novel but fine: the off-nominal control · 6 Against FARM and REAP `[to run]` · 7 Limitations.

Figures: (1) error over time, failure vs success — `2026-10-24_vla_jepa_surprise` (today: n = 1 failure, exploratory);
(2) caught / FI per monitor `[to run]`; (3) off-nominal successes' error distribution — 2026-10-27 frames (`[to run]`, $0).
