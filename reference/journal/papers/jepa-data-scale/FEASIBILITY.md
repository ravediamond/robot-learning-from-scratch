# Feasibility — "The small-data curve: when does a JEPA latent stop being kinematics?"

WS-106, 2026-11-01. Memo only (template `journal/prompts/2026-10-31_paper_feasibility_TEMPLATE.md`, prompt
`journal/prompts/2026-11-01_paper_jepa-data-scale.md`): nothing was run or trained. Every gate cited here is `trained`
in `journal/papers/jepa-world-model/AUDIT_W1_LOADS.md` (26 trained, 1 random = G11-A, not cited). EXPERIMENTS rows are
cited by their current line in `journal/EXPERIMENTS.md`.

## 1. Claim in one sentence

Trained with the G0 recipe at the same number of gradient steps on 45, 90, ~180, ~810 and ~2,900 SO-101 episodes,
a LeWM-style JEPA's latent rises toward the joints (R², now 0.565 at ~180) while the three uses that failed at ~180
(G1b ranking, §30 action beyond state, W1-A mid-rollout screening) stay at their kinematics floor; the claim is false
if, at any scale, one of those three probes clears both its original bar and its floor with the episode-bootstrap CI.

## 2. Evidence in hand

All at one scale (S3: W1, `so101_bowls_all_trimmed`, 182 episodes, 61,462 frames at 30 fps — EXPERIMENTS line 171).

| gate (line) | date | n | result | bar | verdict | pre-reg. | role in the curve |
|---|---|---|---|---|---|---|---|
| G0 (63) | 10-11 | 25 val ep. | open loop K = 30: 0.31 vs copy-last 0.89; 100 epochs, 12.1 h, $3.30 (+ $2.96 failed) | beats copy-last, ratios ≥ 1.3 | pass | same-commit | the recipe and the cost unit |
| G1b W1 (65) | 10-12 | 703 anchors | top-1 best 0.60 (d = 20–30) | ≥ 0.8 | fail | bar-first | probe 2 at S3 |
| G1 probes (66) | 10-12 | 25 val ep. | arm-pose R² W1 0.57, W2 0.92 | > 0.57 | W2 pass | same-commit | probe 1, first version |
| G1 diagnosis (69) | 10-12/13 | 703 anchors | action-only top-1 0.68–0.90, regret ≈ 0; W1 regret 0.26 | exploratory | kinematics | — | probe 2's floor |
| G10 (82) | 10-17 | 69 rollouts | step 1 readout at rest AUROC W1 1.000, W2 0.976 | ≥ 0.90 | pass (retrospective) | same-commit | probe 5 (must not fall) |
| W1-A (98) | 10-19 | 50 (8 fail) | L2 surprise 2/8 at 2/42 (AUROC 0.86) | ≥ 5/8 at ≤ 1/20 | fail | bar-first | probe 4 at S3 |
| W2′ (113) | 10-23 | 25 val ep. | reproduces W2: R² 0.9193; $0.65 | reproduce W2 | reproduced | — | frozen-encoder control and its cost |
| LeJEPA gate 0 + corr. (120, 123) | 10-27/28 | 157 / 25 ep., 8,447 val frames | W1 mean R² 0.565, anisotropy 0.639; W2′ 0.867 / 0.301 | R² < 0.80; ≥ 0.20 | B1–B3 met | bar-first | probe 1 (corrected protocol, the split) |
| HWM gate 0 (124) | 10-28 | 157 / 25 ep. | no crossover; verdict not proven, "open confound: ~0.6 h of one task" | crossover in [1, 2] s | H1, H2 fail | bar-first | the owner's own data-scale confound |
| §30 (128) | 10-26 | teleop val + 69 | (ii)−max = +0.748, but (v)−(iv) arm +0.007, random-init 1.254 vs W1 1.180 | three-part rule (`35f4600`) | not proven | same-commit | probe 3 at S3 |

**Counts.** 10 rows; at S3: 3 positives (G0, G10, LeJEPA as predicted), 4 fails / not proven that are the curve's
y-axis (G1b, §30, W1-A; HWM as context). Pre-registration: 5 bar-first, 4 same-commit (WS-091 §2 classes). Blind
labels: none. Statistic: W1-A and later use the episode bootstrap (2,000); G1b, G1 probes, G10 report point estimates —
recomputable from `projects/vla-jepa-world-model/wm_train/results/` ($0). No other scale point exists: every number
above is one model, one data size.

**Corrections to the pre-written design** (the memo refines; a PROTOCOL.md freezes; no bar is changed here):

1. **§30's bar is three-part** (`35f4600`: (ii)−max ≥ 0.10 with CI > 0 **and** (v)−(iv) ≥ 0.02 arm or ≥ 5 pt
   gripper **and** random-init gain < half W1's). The prompt's "Δ = (ii) − max((i), (iii)), bar 0.10" keeps only the
   conjunct W1 already passes (+0.748): the curve would show a "pass" at every scale. The protocol must carry the
   original rule.
2. **G1b's floor is above its bar**: action-only top-1 0.68–0.90 (line 69) vs the bar 0.8. A model matching the
   commanded joints passes the bar without leaving the floor. Bar and floor must both be reported; which one the
   headline uses is the owner's decision (`needs-from-owner`).
3. **Nested scales.** As written, S4 is the 653 cubes-bowls episodes alone — not a superset of S3, 21 tasks instead
   of one, so S3 → S4 changes the task as well as the size. 157 (S3 train) + 653 + 2,099 = 2,909, the item's
   "2,900": recommend S4 = S3-train ∪ cubes-bowls, S5 = S4 ∪ ArmnetBench, with the 25 bowls val episodes
   (`split.json`, 157 / 25) held out at every scale.
4. **S5 is a second rig**, not "one rig": ArmnetBench is another lab's SO-101 (front + top 1024×576, wrist 1280×720,
   20 fps, joints in degrees — `data/community/REPORT.md` line 118) vs ours (front_top + wrist, 30 Hz,
   `config/lewm_mv_so101.yaml`). S5 measures data and rig together; the paper says so.
5. **"SIGReg off" is not "the plain LeWM loss"**: LeWM's loss is prediction + SIGReg (G0 log: SIGReg 20.1 → 2.9,
   `wm_train/README.md`). SIGReg off is the collapse ablation. The second recipe is already in the design: W2′
   (frozen DINOv2 + predictor, the DINO-WM recipe).
6. **Frozen encoders give flat lines on encoder-only probes.** For W2′ and the random frozen encoder, R², §30 and the
   rest readout (G10 step 1) read encoder latents only, which do not change with scale. Only the predictor probes
   (G1b, W1-A surprise) vary. Those controls need training only for the predictor probes.
7. **Seed and S3.** G0's seed is 3072 (`config/lewm_mv_so101.yaml`), not 0; S3 = the existing W1 @ `d0336db6` (no
   retrain) if its split is the nested one.
8. **Fixed steps at S1** = 81,100 steps (811 / epoch × 100, `wm_train/README.md`) on ~39 train demos ≈ 400 epochs:
   overfitting is likely. The protocol needs a checkpoint rule (final vs best-val), not set here.
9. **G10's 69 rollouts**: WS-091 §2 records them as seen (unlabelled) in training; G0's config names only
   `so101_bowls_all_trimmed`. To resolve before the protocol, and the same rule at every scale.

## 3. Gaps, costed

Cost unit: G0, 100 epochs = 81,100 steps, 12.1 h, $3.30 (EXPERIMENTS line 63; README "Faithful 100-epoch run"),
i.e. ≈ $0.27/GPU-h. Primary budget (b): fixed steps = S3's, so every W1 training costs one G0.

| gap | what | resource | queued? |
|---|---|---|---|
| 1. W1 trainings, option (b) | S1 45, S2 90 (seed-0 subsets of the 157 train), S4, S5a (all 2,099 rollouts), S5b (515 successful rollouts: 915 successes − 400 teleop, REPORT lines 139–150); S3 = W1 | 5 × $3.30 = **$16.50** vast, ~12 h each (parallel) | new |
| 1′. option (a) at S4 | fixed 100 epochs: S4 ≈ 53k + 222k frames ≈ 5.2× S3's train frames → ~63 h | **≈ $17** (optional) | new |
| 2. probe battery | one `probes.py`: R² (corrected protocol), G1b + action-only, §30 three-part, W1-A on the 50, G10 step 1; bootstrap 2,000 | $0; ~8 Orin h per W1 checkpoint (from WS-091 §3: three probes ~6 h) × 7 + ~4 h × 8 predictor-only controls ≈ **~90 Orin h (~4 days)** | new |
| 3. seeds | one extra seed at S1 and S5a (prompt §3) / two each (design) | **$6.60** / $13.20 | new |
| 4. W2′ per scale | predictor only, frozen DINOv2-S; S3 exists (`w2prime` @ `b5bd6036`) | 5 × $0.65 = **$3.25** | new |
| 5. random frozen encoder | predictor only, at S1, S3, S5a (predictor probes only, correction 6) | 3 × $0.65 = **$1.95** | new |
| 6. ArmnetBench preprocessing | 60.5 GB (REPORT line 118) > 8.7 GB free on the Orin: download and transcode to 224 H.264 on a vast CPU box (~2.8 GB JPEG after, REPORT line 200); camera map, 20 → 30 Hz stride rule, degrees → our joint units: a FACTS.json entry | ~4 vast CPU h, **≤ $1.5** (estimate); ~0.5 Orin day for the mapping check | new |
| 7. protocol | the nine corrections above + bars copied from the original protocols; the G1b headline (bar or floor) and the checkpoint rule from the owner | owner ~1 h | new |
| 8. second recipe | not needed: W2′ is it (correction 5); SIGReg off at S3 only as a collapse check | $3.30 optional | — |
| **total (primary)** | 1 + 2 + 3 (one seed) + 4 + 5 + 6 | **≈ $30 vast** (+ ~$10 contingency: G0's failed attempts were $2.96 on $3.30), **~4.5 Orin days, 0 arm days** | |

The item's $15–20 covers gap 1 only. With every optional line: ≈ $57.

Frame counts per scale (the paper's x-axis): S3 61,462 (182 ep., line 171; ~53k in the 157 train, derived); cubes-bowls
222k (from `journal/prompts/2026-10-30_system2_g05_subtasks.md`; to confirm from the dataset meta); ArmnetBench rollouts
14.0 h at 20 fps ≈ 1.01 M frames (REPORT line 133, derived). S5a ≈ 1.28 M frames, ~24× S3.

## 4. Related work

Smallest data point per paper, from the PDFs on disk (`~/workspace/papers/`).

| paper | what it showed | smallest data point | different here |
|---|---|---|---|
| DINO-WM 2411.04983 | frozen DINOv2 + predictor plans in six sim envs; App. A.4.1 Table 5: PushT success 0.08 / 0.48 / 0.72 / 0.88 / 0.92 at n = 200 / 1,000 / 5,000 / 10,000 / 18,500 trajectories | 200 trajectories (PushT) | the closest prior curve: sim, planning success as y; here real arm, the failed uses as y, our W2′ is its recipe |
| LeWorldModel 2603.19312 | end-to-end JEPA with SIGReg from pixels; Push-T and sim | not in the mirror copy (ends at p. 10; App. E, datasets, missing) | the recipe; real data, five sizes |
| LeJEPA identifiability 2605.26379 | SIGReg latents identify state up to anisotropy | 10,000 episodes × 201 steps (trajectory condition, App.) | R² vs data on a real arm; ours starts at 45 |
| RoboJEPA 2610.10515 | 8B multi-embodiment JEPA; scaling law L(C) in compute, models 22M–2B fitted, ~2.87M trajectories, 12 platforms | the compute sweep holds the mixture; the related-work cites an 8,000-trajectory DROID predictor (Terver et al.) | data at fixed compute, ~1,000× smaller, task probes not ℓ1 |
| V-JEPA 2.1 2603.14482 | dense video JEPA features; action-conditioned post-training on DROID | DROID raw (no smaller point found in the text) | we cannot run it (526 ms on the Orin, line 71–73) |
| SaiVLA-JEPA 2610.38508 | future-latent head +0.03 on control, 10,030 LIBERO-Plus episodes per policy | 2,519 LIBERO-10 episodes (one suite) | sim; agrees that latent prediction is not what control needs |
| Robot Data Factory 2609.16705, Ego4WAM 2609.40341 | data-scaling studies for robot learning (on disk, `scaling` / `dataset` tags) | not read for this memo | — |
| HIJP 2610.23615, G-JEPA 2610.28483, planning-limits-jepa 2609.39235 | **not on disk, not read here** | — | to add to Drive before the related-work section |

## 5. Objections

**(a) "2,900 episodes is still nothing."** True against RoboJEPA's ~2.87M trajectories. The claim is the shape of the
curve over ~1.8 decades (45 → 2,909 episodes, ~15k → ~1.3M frames, derived from the counts in §3) on a $200 arm, stated as the condition; DINO-WM's own
curve (Table 5) moves from 0.08 to 0.48 between 200 and 1,000 trajectories, so a real effect at this range is not
implausible.

**(b) "Rollouts are not demos, and S5 is another rig."** S5a vs S5b (all vs successful rollouts) and S4 vs S5 separate
the two; the paper reports both and says S5 changes rig (correction 4).

**(c) "One recipe."** W2′ is the second recipe at every scale for the predictor probes; for encoder probes the frozen
encoder is a flat line by construction (correction 6), which is itself the comparison.

**(d) "The probes measure the wrong thing."** They are the register's own failed uses, each a published paper's claim,
with the original bars; the paper says so.

## 6. Venue

Primary: a world-model or robot-learning workshop (ICLR 2027 or NeurIPS 2027; dates to confirm on the workshop pages,
4 pages, double-blind). Fallback: TMLR (rolling). **Fold vs split:** by default this is §6 of WS-091's paper (its
"scale point [to run]" becomes a curve); a separate paper only if a probe leaves its floor. arXiv first: after the
curve. Endorsement: no contacts list in the repo (as WS-091).

## 7. Minimal vs full paper

4 pages (or WS-091 §6): primary curve, five panels, W1 / W2′ / floors, gaps 1, 2, 4, 6 (≈ $21). 8 pages: adds seeds
(gap 3), the random-encoder floor (gap 5), option (a) at S4 and the S5a / S5b split discussion (≈ $47 with 1′).

## 8. Risk of a null result

The expected outcome is the null: R² rises, probes 2–4 stay at the floor. That is WS-091's thesis with a curve, and the
paper the item's `why` already accepts ("a flat curve is still the paper"). The other risk: R² does not rise either
(the latent is not even kinematics-limited by data). Then the result is "the recipe, not the data, is the limit at
this range" — weaker, still a figure. A probe leaving its floor is the better paper (split, correction 2 decides
whether G1b counts).

## 9. Recommendation

**go-after-X**, X = WS-091's go/no-go (this curve is its scale point) and a PROTOCOL.md that resolves the nine
corrections in §2, with the owner's two decisions (G1b headline: bar or floor; checkpoint rule). Cost ≈ $30 vast
(+ $10 contingency), ~4.5 Orin days, 0 arm days. Earliest: protocol 1 day, trainings in parallel ~1 day, probes ~4
Orin days, writing ~1 week → a draft ~3 weeks after the go (~2026-11-22 if the go is 2026-11-02).

## 10. Skeleton

Titles: (1) *The small-data curve: when does a JEPA latent stop being kinematics?* (2) *From 45 to 2,900 episodes: a
JEPA world model on a $200 arm learns the arm first* (3) *Data, not architecture? Five scales of one JEPA world model*.

Abstract (≤ 150 words): A LeWM-style JEPA world model trained on 180 demonstrations of one SO-101 task recovers the
arm's joints (R² 0.565) but fails every pre-registered use that needs more than the pose. Is that the data? We train
the same recipe at equal compute on 45, 90, 180, 810 and 2,909 episodes — the last adding 2,099 public rollouts from
another SO-101 rig — and rerun the failed uses at each point with their original bars: ranking candidate action
chunks, action information beyond the state, and mid-rollout failure screening. A frozen DINOv2 encoder and the
commanded joints are the floors. R² rises from [to run] to [to run]; ranking, beyond-state action and screening
[leave / do not leave] their floors by 2,909 episodes ([to run]). Reading a finished episode's outcome stays at
[to run]. At the data a single owner can collect, the curve, not the architecture, says what the latent carries.

Sections: 1 Intro · 2 Recipe, rig, the five scales (frames, not episodes) · 3 The probe battery, bars and floors ·
4 The curve · 5 Rollouts vs demos, second rig (S5a / S5b) · 6 Related work · 7 Limitations.

Figures: (1) five panels, x = log frames, W1 / W2′ / floors with bootstrap bands — S3 point today from
`projects/lejepa-identifiability/results/gate0_corrected_2026-10-28.json`, `wm_train/results/2026-10-12_g1b_range_w1.json`,
`shared/gate/data/2026-10-19_w1_screening/`, `shared/gate/data/2026-10-24_inverse_dynamics/`; the rest [to run];
(2) DINO-WM Table 5 replotted beside ours (from the PDF); (3) frames per scale and source (today, from the dataset
meta).
