# Feasibility — A Platonic test on a robot: do the top and wrist cameras learn the same latent up to a rotation, and does an action-conditioned predictor break it? (WS-107, 2026-11-01)

Paper read in full from the mirror: `~/workspace/papers/2610.39302v1-platonic-jepa.pdf` (Balestriero & LeCun,
"Proving and Improving the Platonic Representation Hypothesis with JEPAs"; text extract `runs/WS-107/paper_raw.txt`).
Template: `journal/prompts/2026-10-31_paper_feasibility_TEMPLATE.md`. Nothing run; $0. Only gates marked `trained` in
`journal/papers/jepa-world-model/AUDIT_W1_LOADS.md` are cited.

## 1. Claim in one sentence

Two JEPAs trained separately on the SO-101 top camera and wrist camera, with the same temporal positives, learn
embeddings related by one orthogonal map: mutual 10-NN overlap between them on held-out paired frames exceeds the
bad-view control with a 95 % episode-bootstrap CI (B1), and switching on W1's action-conditioned predictor moves
that overlap by an amount whose CI excludes 0 (B2, direction pre-registered as "lower"; see §5 Q3).

## 2. Evidence in hand

| id | date | n | result | CI | status | anchor |
|---|---|---|---|---|---|---|
| G0 (W1, multi-view LeWM, 100 epochs) | 10-11 | 192 held-out windows, 25 val episodes | K = 30 error 0.31 vs copy-last 0.89; effective rank 24–30 / 192; $3.30 | none | pass, `trained` | EXPERIMENTS row G0; `projects/vla-jepa-world-model/wm_train/README.md` §Conclusion (2026-10-11) |
| W1 per-view rank (same run) | 10-11 | all val frames | encoder CLS eff. rank: front 15.3, wrist 10.1 (val); fused z 24.0 | none | measured, `trained` | `wm_train/README.md` rank table (lines 197–204) |
| G1 probes | 10-12 | val split | arm-pose R²: W1 0.57, W2 (frozen DINOv2-S) 0.92 | none | W2 pass, `trained` | EXPERIMENTS row G1 probes |
| W2′ re-train | 10-23 | val split | arm-pose R² 0.9193, reproduces W2 | none | reproduced, `trained` | `wm_train/results/2026-10-23_g1_final_w2prime_best.json` |
| LeJEPA identifiability gate 0 (corrected) | 10-28 | 157 train / 25 val episodes | W1 joint R² 0.565, anisotropy 0.639; W2′ 0.867 / 0.301 | none | B1–B3 met, `trained` | `projects/lejepa-identifiability/results/gate0_corrected_2026-10-28.json` |
| G9c top-camera judge (HSV masks) | 10-16 | 88 rollouts | accuracy 86.4 % [78, 93], AUROC 0.919 [0.87, 0.96] | yes | fail (pre-registered) | `projects/vla-jepa-world-model/judge/results/g9c/report.md` |
| Wrist blank, proxies (SmolVLA via vla.cpp, not W1) | 10-23 | 69 rollouts | chunk error 2.28° → 3.48°, blank worse on 54/69 | none | measured | `projects/so101-multitask-smolvla/data/2026-10-23_wrist_blank/README.md` |
| G-lat-2 (SmolVLA server split) | 10-20 | 400 chunks | total 219.0 ms, vision 151.0 ms (both cameras) | none | pass (own bar) | `shared/gate/data/2026-10-20_glat2/README.md` |

Datasets: `so101_bowls_all` 182 episodes, rollouts 69 (44 success / 25 failure) — `data/FACTS.json`.

- **Pre-registered:** G9c (bar before run), LeJEPA gate 0 (`ad40675`, `e6ba2fe`), wrist blank (`46ebb00`). G0 and G1
  bars were in their protocols; no hash recorded in the rows.
- **Blind-labelled:** none bears on alignment.
- **Current statistic (episode bootstrap, 2,000 resamples):** G9c only. None of the rows measures alignment; no row
  needs recomputation for this claim.
- **What none of them shows:** any cross-camera alignment number. W1's two views share one ViT-Tiny encoder and are
  fused by a mean before the predictor (`wm_train/train_lewm_mv.py` lines 2–3, 117–125), so W1 is not "trained
  separately" in the theorem's sense; it is the joint reference, not a measurement.
- **Two corrections to the prompt's framing.** (i) The 151 ms is SmolVLA's vision for *both* cameras (G-lat-2), not
  the wrist encode; the wrist's share is **missing** (one Orin timing run, $0). (ii) The wrist-blank result is on
  SmolVLA, not on any JEPA; it shows the policy uses the wrist, not that the wrist latent carries scene information.

## 3. Gaps, costed

Costing from G0: 100 epochs multi-view = $3.30 (9.2 h at $0.28/h measured, `wm_train/README.md` line 120). A
single-view run encodes half the images per step; ≤ $3 each is used as the ceiling.

| # | gap | resource | queued? |
|---|---|---|---|
| 0 | encoder-only training mode: identity predictor, positives = (frame_t, frame_t+Δ) of one camera, SIGReg; the per-camera LeWM config exists (`config/lewm_so101.yaml`, predictor-on) | ~0.5 Orin day code + 5-epoch smoke on vast ~$0.3 | new |
| 1 | E_top, E_wrist (encoder-only, temporal positives — the paper's own video good views, MSR-VTT Fig. 8) | 2 × ≤ $3 vast | new |
| 2 | P_top, P_wrist (same, W1's action-conditioned predictor on) | 2 × ≤ $3 vast | new |
| 3 | bad-view control: per camera, positives = two crops of one frame (paper Fig. 8 design) — **refinement**, see note | 2 × ≤ $3 vast (prompt's single E_crop pair: 2 × ≤ $3 too) | new |
| 4 | upper anchor: E_top seed 2 | ≤ $3 vast | new |
| 5 | optional E_top×wrist (cross-camera positives, a CLIP-like joint model) | ≤ $3 vast | new, cut from minimal |
| 6 | alignment script: mutual 10-NN (Huh et al. definition, k = 10) + Procrustes residual fitted on half, scored on the other half; whitened; episode bootstrap 2,000 | ~0.5 Orin day, $0 | new |
| 7 | F_dino (frozen DINOv2-S, top vs wrist) and W1 per-view CLS (shared encoder) on the same pairs | Orin CPU/GPU ~1 h, $0 | new |
| 8 | ρ estimate (paper's way, App. "Nuisance on real encoders"): needs several observations of one state; use still-arm, still-scene windows as repeated observations (C_f = cov of window means minus within-window cov / n) | ~0.5 Orin day, $0 | new |
| 9 | scene probe: bowl centroids from G9c masks + gripper state, R² per latent (the prompt's ρ proxy; kept as a separate descriptive number) | ~0.5 Orin day, $0 | new |
| 10 | wrist encode share of the 151 ms (SmolVLA vision, one camera vs two) | 1 Orin timing run, $0 | new |
| 11 | PROTOCOL.md frozen (bars B1, B2; B3 descriptive) before any training | owner review | new (frozen-protocol skill) |
| | **total** | **vast: minimal 6 trainings ≈ $18 (+ $0.3 smoke); with E_top×wrist ≈ $21; ~3 Orin days; 0 arm days; 0 teleop; 0 labelling hours** | |

Note on gap 3: the prompt's lower anchor is "two random crops of the top image → E_crop1 / E_crop2". The paper's
image-side test (§3.3, Fig. 8) instead trains a crops-of-one-frame (bad-view) JEPA on each side and compares it with
the frames-of-a-clip (good-view) JEPA. The Fig. 8 form tests Proposition 3 on the same two cameras and keeps B1 a
like-for-like comparison. Which form control 1 takes is a protocol field: **owner decision**, not changed here.

Note on the metric: within one episode at 5 fps, neighbouring frames are near-duplicates in both cameras, so a plain
mutual 10-NN rewards temporal smoothness rather than shared scene content. The protocol should state whether
neighbours from the same episode (or within ±N frames) are excluded; the paper's WIT-1024 pairs are independent
images, so exclusion is the closer match. **Owner decision.**

## 4. Related work

| id | what it showed | what differs here |
|---|---|---|
| 2610.39302 Platonic JEPAs | Theorem 1: shared positives → same embedding up to one rotation; Thm 2 bound by kernel mismatch / eigengap; Thm 3 alignment error ≥ 2k(1 − min ρ); Prop 3 crops = bad views; ρ for k = 32: .804 good-view vs .655 bad-view vs .608 next-token (§3.1, Table 6); MSR-VTT frames-of-a-clip vs crops (Fig. 8) | real robot, two physical cameras, same timestamps; adds the action-conditioned predictor the analysis excludes (§2.1: "a learned predictor, often conditioned on an action, which our analysis does not cover") |
| 2605.26379 LeJEPA identifiability | SIGReg + alignment give linear identifiability of latent world variables | our gate 0 measured it on W1 (R² 0.565); here the question is cross-view, not latent-to-state |
| 2603.19312 LeWorldModel | end-to-end JEPA from pixels, SIGReg, next-embedding prediction | W1's recipe; we switch its predictor off and on |
| 2608.13438 ContactGuard | action-conditioned multi-view latent world model as a monitor | W1's architecture (shared encoder, mean over views); never measured cross-view alignment |
| 2610.00727 CF-JEPA | splits latents into controllable and uncontrollable factors | its split is the natural reading of "what the predictor adds"; not about cross-view alignment |
| 2411.04983 DINO-WM | world model on frozen DINOv2 patches | our F_dino row uses the same frozen encoder |
| 2506.09985 V-JEPA 2; 2603.14482 V-JEPA 2.1 | video JEPA, robot planning; dense features, multi-modal tokenizers | train on many views jointly; no separately trained per-camera alignment |
| 2608.06799, 2609.03565 physical-grounding JEPA | not in the mirror, not read here | — |
| Huh et al. 2024, Platonic Representation Hypothesis (arXiv 2405.07987) | mutual k-NN across scaled vision/language models | not in the mirror, not read here; the metric definition is taken from 2610.39302's use of it |

Of the papers on disk, none reports alignment between two robot cameras' separately trained encoders.

## 5. Objections

**Q1 "Two cameras of one table is a toy."** It is Theorem 1's setting: two renderings of one world, with the same
positive pairs — with temporal positives both cameras' pairs are (z_t, z_t+Δ) at identical timestamps, so π_top =
π_wrist exactly, which the paper's image–text experiments cannot arrange. What the rig breaks is injectivity: the
wrist sees part of the table, so many states give one wrist image (the paper's "Known limitations": "real modalities
are lossy"). The honest framing: the first test with exactly shared positives and a known lossy view.

**Q2 "What is nuisance here?"** The prompt's nuisance proxy (wrist pose from joints) is not nuisance in the paper's
sense: §2.3 defines ν as drawn independently of the state, and the arm pose is part of the state (the top camera sees
the arm). Under the theorem, wrist pose should *align*. The theorem's nuisance on this rig is sensor noise, blur and
lighting per camera; the wrist's loss of scene is a kernel-mismatch / non-injectivity term (Thm 2), not ρ. Gap 8
measures ρ the paper's way; gap 9 keeps the scene probe as a descriptive number. Bars are unchanged.

**Q3 "Why should the predictor lower alignment?"** The paper makes no prediction for action-conditioned predictors;
it excludes them (§2.1). The prompt cites Proposition 3 for B2, but Proposition 3 is about corrupted positives (crops),
not predictors. An argument the other way exists: action conditioning rewards encoding what actions move (the arm),
which both cameras see, so alignment could rise. B2's pre-registered direction stands as written; the memo flags
that its rationale is not from the paper and that a result in the other direction is also informative. **Owner: keep
B2 one-sided "lower", or make it two-sided, before the protocol is frozen.**

**Q4 "Alignment depends on the probe."** Use the paper's metric unchanged (mutual 10-NN, k = 10) plus the Procrustes
residual of Proposition 2 (the theory's own quantity), both on whitened embeddings, rotation fitted on half the
episodes.

**Q5 "Tiny, under-trained encoders."** ViT-Tiny, 192 dims, 182 episodes. One scale here; the data-scale item supplies
larger points. Upper anchor (seed 2) and F_dino bound the range on the same pairs.

**Q6 "SIGReg is not covered."** The paper proves uniqueness up to rotation only for second-order (whitened)
anti-collapse and leaves SIGReg's full-distribution constraint as an assumption (§2.1 after Prop. 1). Our encoders use
SIGReg; we whiten before measuring, as the paper does, and state the assumption.

**Q7 "So what for robots?"** If Q·E_top predicts E_wrist at frames where the wrist sees the bowl, a wrist encode could
be skipped there. The latency at stake is **missing**: 151 ms is both cameras of SmolVLA (G-lat-2); gap 10 measures
the wrist share. Listed as a follow-up for the edge-inference paper (`journal/papers/edge-inference/FEASIBILITY.md`),
not run here.

## 6. Venue

Primary: a representation-learning or world-model workshop at ICLR 2027 (4 pages, double-blind; dates to confirm on
the workshop pages). Fallback: TMLR (rolling, no page limit, negative results in scope). arXiv first: only after B1/B2
exist; a theory paper this recent (2026-10-29) makes a fast, short empirical note the natural form. Endorsement
(cs.LG / cs.RO first submission): no contacts list in the repo (same finding as
`journal/papers/jepa-world-model/FEASIBILITY.md` §6); nobody named.

## 7. Minimal paper vs full paper

- **4-page workshop:** E_top, E_wrist, P_top, P_wrist, bad-view control, seed anchor, F_dino; mutual 10-NN +
  Procrustes with episode CIs; B1, B2; one occlusion-bin figure (B3). Needs gaps 0–4, 6, 7, 11 (≈ $18).
- **8-page:** adds ρ the paper's way (gap 8) with Theorem 3's lower bound checked against the measured error (as the
  paper's Table 6 does), E_top×wrist (gap 5), the rollout set (69, out of distribution) as a second test set, the
  per-direction alignment curve (paper Fig. 6), the wrist-share latency (gap 10) and a Q·top substitution test
  offline. ≈ $21 + ~2 more Orin days.

## 8. Risk of a null result

- **B1 fails** (top–wrist no better than bad views): reads as "a lossy wrist view breaks the theorem's injectivity
  enough to erase alignment at this scale" — a limitation the paper names but does not measure. Publishable as a short
  negative note if the upper anchor and F_dino show the metric works on these pairs.
- **B2 fails** (predictor changes nothing, or raises alignment): the first measurement on the case the paper
  excludes; either direction is a result (Q3).
- **Uninformative outcome:** the upper anchor (two seeds, same camera) itself low — the encoders are not unique up to
  rotation at this scale and nothing can be read. This is the main risk; a 5-epoch pilot cannot settle it before the
  protocol without producing an unregistered number.

## 9. Recommendation

**go-after-protocol:** freeze PROTOCOL.md with the three owner decisions (control-1 form, same-episode neighbour
exclusion, B2 one- or two-sided), then run. Cost ≈ $18 vast (minimal; ≈ $21 with E_top×wrist), ~3 Orin days, 0 arm
days. With one Orin and no arm need: protocol + code ~1 week, trainings ~2 days, analysis ~2 days; earliest
realistic submission about three weeks after the owner's go, subject to the workshop deadline (to confirm).

## 10. Skeleton

Titles: (a) "A Platonic Test on a Robot: Two Cameras, One Rotation"; (b) "Do a Robot's Cameras Learn the Same World?
Testing the Platonic Representation Theorem on an SO-101"; (c) "Action-Conditioned Predictors and the Platonic
Representation: A Two-Camera Test".

Abstract (≤ 150 words): Balestriero and LeCun prove that JEPAs trained separately on two renderings of one world, with
shared positives, learn the same embedding up to a rotation, and exclude action-conditioned predictors. A robot with
two cameras is that setting with exactly shared temporal positives and one lossy view. On an SO-101 with a top and a
wrist camera (182 episodes), we train single-camera JEPAs with W1's recipe, predictor off and on. Top and wrist
embeddings reach mutual 10-NN `[to run]` against `[to run]` for crop-trained bad views and `[to run]` for two seeds of
one camera (95 % episode CI); the Procrustes residual is `[to run]`. Switching on the action-conditioned predictor
changes alignment by `[to run]`. Alignment falls with wrist occlusion `[to run]`. We discuss when the wrist encode can
be replaced by a rotated top embedding.

Sections: 1 Intro (theorem, the excluded case, the rig); 2 Setting (world = scene + arm, cameras as renderings,
temporal positives = shared π, the wrist as a lossy rendering); 3 Models and metric (6–7 encoders, mutual 10-NN,
Procrustes, whitening, episode bootstrap); 4 Results (B1, B2, B3); 5 Nuisance and ρ (8-page only); 6 Practical
reading (wrist substitution, latency follow-up); 7 Limitations (SIGReg, one scale, one task).

Figures: F1 rig and the two views — `[to run]` (frames exist in `so101_bowls_all`); F2 alignment ladder (bad view,
top–wrist, predictor-on, seed anchor, F_dino) with CIs — `[to run]`; F3 alignment vs wrist occlusion bin, occlusion
from G9c masks — `[to run]`; F4 (8-page) per-direction rotated correlation, as paper Fig. 6 — `[to run]`.
