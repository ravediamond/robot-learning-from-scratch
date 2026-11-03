# Feasibility — "Distilling a big video JEPA into an Orin-sized student": does a pretrained teacher's knowledge survive the shrink on 653 episodes, at edge latency?

WS-110, 2026-11-01. Memo only (template `journal/prompts/2026-10-31_paper_feasibility_TEMPLATE.md`, prompt
`journal/prompts/2026-11-01_paper_jepa-distill-edge.md`): nothing was run. Only gates marked `trained` in
`journal/papers/jepa-world-model/AUDIT_W1_LOADS.md` are cited. EXPERIMENTS rows are cited by line
(extract: `runs/WS-110/exp_rows.md`).

**Correction to the prompt's design.** The teacher already on the Orin is **V-JEPA 2** ViT-L, not V-JEPA 2.1:
VLA-JEPA bundles `facebook/vjepa2-vitl-fpc64-256` as `model.video_encoder.*`, 587 tensors, 325.97 M parameters, BF16,
patch embedding `[1024, 3, 2, 16, 16]` (tubelet 2, patch 16), 8 frames per view
(`shared/gate/data/2026-10-24_vla_jepa_surprise/README.md` line 30). A V-JEPA 2.1 ViT-L (300 M) is a separate download.
Which one is the teacher is a protocol field (owner, §3 row 8).

## 1. Claim in one sentence

A W1-sized student (192-d latent, the architecture that runs a dual-camera check in 12.78 ms on the Orin) trained
on 653 SO-101 episodes to regress a 326 M video-JEPA teacher's latents beats the same student trained from scratch on
G1b top-1 or the §30 beyond-state term, with an episode-bootstrap CI > 0, and beats a random-teacher control on the
same probe, at ≤ 2× W1's latency; false if no distilled variant clears the scratch and random-teacher rows (bars B1,
B2 of the prompt's design, frozen later in a PROTOCOL.md, not here).

## 2. Evidence in hand

| gate (EXPERIMENTS line) | date | n | result | bar | verdict | pre-reg. | bears on |
|---|---|---|---|---|---|---|---|
| G0 (63) | 10-11 | 25 val ep. | W1 open loop K = 30 0.31 vs copy-last 0.89 | beats copy-last; ratios ≥ 1.3 | pass | same-commit | the student's recipe and cost ($3.30) |
| G1b W1 (65) | 10-12 | 703 anchors | top-1 best 0.60, never ≥ 0.8 | top-1 ≥ 0.8 | fail | bar-first | B1 probe 1, scratch value |
| G1 probes (66) | 10-12 | 25 val ep. | arm-pose R² W1 0.57, W2 (frozen DINOv2-S) 0.92 | > 0.57 | W2 pass | same-commit | pretrained features read pose better |
| G2 Mac (67) | 10-12 | — | one decision W1 108 ms, W2 1814 ms (Mac MPS) | before next chunk | W1 ok, W2 too slow | — | why a frozen big encoder is not the edge answer |
| RoboJEPA Gate B (71) | 10-13 | — | V-JEPA 2.1-G encoder 2 views 526 ms (1 view 398 ms); predictor step 100M 252 ms | p50 ≤ 200 ms | no-go on device | — | the route this item replaces |
| RoboJEPA Gate B′ (72) | 10-13 | — | encoder 1025 ms on MPS | same | no-go | — | off-board host must be CUDA |
| RoboJEPA Gate A (73) | 10-13 | 30–60 clips | 9-step L1 below copy-last: 300M 11.8 %, 100M 10.5 %, 1B 15.8 % [14.6, 17.0] (30 clips); below shuffled 8.9–10.4 % | ≥ 15 % (300M primary); ≥ 5 % | no-go primary | — | the teacher class is weak zero-shot here |
| G11-A (84, 85) | 10-18 | — | check latency mean 12.78 ms, p95 12.87 ms | < 20 ms | pass; **timing valid, latents void** (random encoder, same architecture) | — | the B2 reference |
| W2′ (113) | 10-23 | 25 val ep. | pose R² 0.9193, gripper AUROC 0.9994, open loop / copy 0.633 | reproduce W2 | reproduced | owner | the frozen-pretrained control, exists |
| LeJEPA gate 0, corrected (120, 201) | 10-27/29 | 8,447 val frames | W1 mean R² 0.565, anisotropy 0.639; DINOv2-S 0.867 / 0.301; G1b W1 0.6046 vs W2′ 0.6458 | R² < 0.80; ≥ 0.20 | B1–B3 met | bar-first | +0.30 R² from pretraining moved G1b by 0.04 |
| §30 inverse dynamics (128) | 10-26 | teleop val + 69 | beyond state (v)−(iv) arm +0.007 [0.003, 0.011]; random-init W1 gain 1.254 vs W1 1.180; MLP latents + joints worse than joints (W1 −0.319, W2′ −1.340) | ≥ 0.02; random-init < half | not proven | bar-first (`35f4600`) | B1 probe 2, and the reason for a random-teacher control |
| VLA-JEPA §29 step 0 (README above) | 10-25 | 50 calls | full check (Qwen + V-JEPA 2 × 2 views × 8 frames + predictor) p50 1973 ms at 25 W | — | feasibility | — | the teacher is on disk and loads (739/739) |

**Counts.** 12 rows: 4 pass / reproduced (G0, G1 W2, W2′, G11-A timing), 1 met as predicted (LeJEPA), 1 not proven
(§30), 5 fail / no-go (G1b, Gate A, Gate B, Gate B′, G2 W2), 1 feasibility. No row has ever distilled anything: **the
claim has zero direct evidence**. Pre-registered bar-first: G1b, LeJEPA, §30; G0 and G1 probes same-commit; RoboJEPA
gates predate WS-078's classes. Blind labels: none needed (all probes are against logged joints / expert chunks).
Statistic: LeJEPA and §30 use the episode bootstrap (2,000); G1b and Gate A report point estimates (Gate A's interval
is on 30 clips) — G1b recomputable from `wm_train/results/2026-10-12_g1b_range_w1.json` ($0).

**What the evidence already predicts.** Pretraining moved pose R² by +0.30 (W1 0.565 → DINOv2-S 0.867) and G1b by
+0.04 (0.6046 → 0.6458), both ≪ action-only 0.90 (LeJEPA row). The teacher class beats copy-last by only 10.5–15.8 %
(Gate A). RoboJEPA §3.3: predictors below 300 M "cap at about 0.3 final reward" on its long-horizon task. The prior is
a negative: the student will read the pose better and rank no better.

## 3. Gaps, costed

Data: `so101_cubes_bowls_20261015`, 653 episodes, 222,000 frames, 21 tasks, 30 fps, two 480×640 views
(`~/.cache/huggingface/lerobot/ravediamond/so101_cubes_bowls_20261015/meta/info.json`). At 5 fps: 37,000 frames per
view, 74,000 clips; with the bowls set (61,462 frames → ~10,240 per view) ≈ 94,500 clips (derived). One W1 epoch on
bowls = 811 steps, 5.5 min; 100 epochs ≈ 9.2 h at $0.28/h (`projects/vla-jepa-world-model/wm_train/README.md`
line 120); G0 cost $3.30 with retries (row 63).

| gap | what | resource | queued? |
|---|---|---|---|
| 1. Teacher latents | V-JEPA 2 ViT-L, both views, 8-frame causal clip ending at each 5 fps frame, ~94,500 clips; ≈ 2 × 326 M × 1,024 tokens ≈ 0.67 TFLOP per clip → ~63 PFLOP, < 1 GPU-h of compute, decode-bound → ~2–3 h on G0's box class | **≈ $1** (cap $5); disk: pooled fp16 1024-d ≈ 0.19 GB, 4 × 4 patch grid ≈ 3.1 GB, full token map (1,024 × 1,024 fp16 = 2 MiB per clip) ≈ 190 GB — **not stored**. Orin keeps the pooled file only (8.7 GB free, floor 6 GB); grids stay on vast / a private Hub dataset | new |
| 2. Random-teacher latents | same pass, randomly initialised ViT-L (seed fixed) | **≈ $1**, same disk | new |
| 3. Students D1, D2, D3 | W1 encoder + a 192 → 1024 projector (training only, dropped at inference); D1 cosine + MSE to the pooled latent; D2 D1 + W1's JEPA loss; D3 W1 predictor trained to the teacher's latent at t + k | 3 × one G0 at fixed steps (WS-106 option b) = **$9.90**; at 100 epochs on S4's ~275k frames, ~3.6–4.5× each (~$10 each) | new |
| 4. Random-teacher student | D1's loss on gap 2's latents; per-variant controls (D2, D3) **+$6.60** if the protocol asks | **$3.30** | new |
| 5. Teacher upper reference | frozen V-JEPA 2 grid + W1-sized predictor (DINO-WM's design with the teacher's encoder), probed off-board | **≈ $1** (predictor only, W2′ cost $0.65, row 113) | new |
| 6. Probe battery | WS-106's `probes.py` (R² corrected, G1b + action-only, §30 three-part, W1-A, G10 step 1, bootstrap 2,000) on 4 students + the reference | $0; ~8 Orin h per checkpoint (WS-106 FEASIBILITY §3) → **~40 Orin h** | shared with WS-106 gap 2 (not built) |
| 7. Latency (B2) | each student, 50 calls, 25 W, strict load, G11 protocol; the student is W1's architecture, so B2 ≈ 12.78 ms by construction — measured anyway | $0, ~1 Orin h | new |
| 8. Controls at the same data | W1-scratch at S4 (WS-106 gap 1, $3.30) and W2′ at S4 (WS-106 gap 4, $0.65) — **shared, not counted twice**; S3 rows exist | $0 here | WS-106 |
| 9. Protocol fields for the owner | teacher (V-JEPA 2 on disk vs 2.1 ViT-L download); S4 = cubes-bowls alone or S3-train ∪ cubes-bowls (WS-106 §3 item 3); the G1b headline distance and the §30 term B1 uses; D3's target (teacher latent at t + k from gap 1, $0, vs a teacher-predictor rollout via VLA-JEPA's 157 M predictor, ~$5 more); random-teacher per variant or D1 only; checkpoint rule | owner ~1 h, before PROTOCOL.md | new |
| 10. Off-the-shelf distilled student (optional, exploratory) | V-JEPA 2.1 ships a ViT-B 80 M distilled from ViT-G (Table 11); its Orin encode latency is **missing** | ~0.3 GB download, ~1 Orin h, $0 | new |
| **total** | | **≈ $16 vast** (1 + 1 + 9.90 + 3.30 + 1; +$6.60 with per-variant controls), **~2 Orin days**, 0 arm days, ≤ 0.5 GB on the Orin; depends on WS-106's `probes.py` and S4 controls | |

The prompt's "≈ $10" counted two students; four trainings at G0's recorded cost make it ≈ $16.

## 4. Related work

| paper | what it showed | different here |
|---|---|---|
| V-JEPA 2.1 2603.14482 (read: Table 11, §2.3.6, App. B) | ViT-L 300 M distilled from a frozen ViT-G 2B teacher beats ViT-L from scratch on all seven tasks (SSv2 74.2 → 76.5, K400 84.3 → 86.9, ADE20K 42.0 → 46.7); ViT-B 80 M distilled 74.5 SSv2 | the method is the teacher's own; here the student is W1 (192-d latent; parameter count not looked up), the data 653 robot episodes, the targets robot probes, and latency is measured on an Orin |
| RoboJEPA 2610.10515 (read: §2, §3.1 Table 2, §3.3) | frozen V-JEPA 2.1-G (patch dim 1664) + predictors 22 M–8 B; second-order compute power law; predictors < 300 M cap at ~0.3 final reward; no distillation | Gate A / B on our clips: weak zero-shot, 526 ms; this item tests the shrink RoboJEPA's law does not cover |
| V-JEPA 2-AC 2506.09985 (on disk, not read for this memo) | the teacher family's action-conditioned planning on Franka | our teacher is V-JEPA 2 ViT-L |
| DINO-WM 2411.04983 (read: App. A.9) | frozen DINOv2 patches (14 × 14 × 384) + ~19 M ViT predictor plan zero-shot in six environments | our W2′ is this design; gap 5 is the same design with the teacher's encoder |
| LeWorldModel 2603.19312 (on disk, not re-read) | end-to-end JEPA from pixels with SIGReg | W1 is this recipe; D2 adds the teacher term to it |
| TurboVLA 2607.27205 (read: §5.2, Table 1) | DINOv3 ViT-B / ViT-L encoders, 31.2 ms policy latency, 0.2 B params on an RTX 4090; **no Jetson number** | the prompt's latency reference does not exist in the PDF; W1's 12.78 ms (G11-A) is the reference |
| VLA-JEPA 2602.10098 | JEPA world model with a V-JEPA 2 target inside a VLA | bundles our teacher; its full check 1973 ms on the Orin |
| Platonic JEPAs 2610.39302 (on disk, not read for this memo) | JEPAs converge to shared representations | would predict the student and teacher align; the probes test whether that alignment carries task content |
| Edge distillation of vision encoders (e.g. DeiT 2012.12877, TinyViT 2207.10666, EfficientSAM 2312.00863) | not on disk, **not read here** | to add to Drive before the related-work section |
| WS-071 edge-inference memo (`journal/papers/edge-inference/FEASIBILITY.md`) | on this Orin, vision is 151 of 219 ms of a SmolVLA chunk (G-lat-2) | the latency framing: a big encoder cannot sit in the loop |

## 5. Objections

**(a) "Distillation of a world model is known."** True: V-JEPA 2.1 builds its ViT-L / ViT-B that way (Table 11). The
claim is not the method; it is whether the knowledge reaches robot probes (G1b, §30) in a 192-d student, on
653 episodes and Orin latency, against a random-teacher control the V-JEPA 2.1 table does not have.

**(b) "The teacher was already weak on these clips (Gate A)."** Then the student should be too, and a teacher-reference
row (gap 5) shows how much there was to inherit. If the teacher reference itself does not beat W1-scratch on G1b /
§30, B1 cannot pass and the result reads "the bigger model has nothing beyond kinematics to give on this rig" — a clean
negative that closes the route. Gap 5 should be run first: it is $1 and can make the rest moot.

**(c) "Storage and cost."** Teacher latents once, ≈ $1 each pass; the full token map (≈ 190 GB) is not stored; the
pooled file (0.19 GB) fits the Orin.

**(d) "Why not run the teacher off-board?"** The register's condition is on-device; off-board is WS-071's paper.

**(e) "Regression to any big target regularises."** The random-teacher control (gap 4) is in B1 for that reason;
§30 already showed a random-init W1 matching a trained one (row 128).

## 6. Venue

As WS-091 (`journal/papers/jepa-world-model/FEASIBILITY.md` §6): a world-model / robot-learning workshop (ICLR 2027 or
NeurIPS; 4 pages, double-blind; dates to confirm on the workshop pages), fallback TMLR. Standalone it is thin; the
natural home is **one arm of WS-106's data-scale paper** (same S4 trainings, same `probes.py`). arXiv first: no, not
standalone. Endorsement: no contacts list in the repo.

## 7. Minimal vs full paper

4 pages (or a WS-106 section): teacher reference (gap 5), D1, random teacher, W1-scratch, W2′ at S4 on G1b, §30 and R²,
plus the latency row — needs gaps 1, 2, 3 (D1 only), 4, 5, 7 ≈ $6.30. 8 pages: D2, D3, per-variant random-teacher
controls, the V-JEPA 2.1 ViT-B off-the-shelf row (gap 10), the B3 retention ratio per probe — ≈ $22.

## 8. Risk of a null result

High, and §2 predicts it (pretraining +0.30 R², +0.04 G1b). The negative reads "a big pretrained predictor's knowledge
does not survive the shrink at 653 episodes" — publishable only with the teacher reference beside it (otherwise it is
indistinguishable from "the teacher had nothing"). If the teacher reference also fails B1-style comparisons, the
finding is about the teacher, not distillation, and belongs in WS-091 / WS-106 as one more row.

## 9. Recommendation

**go-after-X**, X = WS-106's S4 controls and `probes.py` exist, then gap 5 first ($1): if the frozen teacher + small
predictor does not beat W1-scratch on G1b or §30 off-board, stop (the route is closed for ~$3 total, write it as a
WS-106 row); if it does, run D1 + random teacher (≈ $5.30) and decide D2 / D3 on that. Full cost ≈ $16 vast
(+$6.60 per-variant controls), ~2 Orin days, 0 arm days. Earliest: ~1 week after WS-106's battery is built (all
offline); as a section of WS-106's paper, not a separate submission, unless B1 passes.

## 10. Skeleton

Titles: (1) *Does a big video JEPA's knowledge survive the shrink to a $200 robot's GPU?*; (2) *Distilling V-JEPA into
a 12 ms world model on 653 demonstrations*; (3) *Teacher, student, scratch: pretrained video latents on a small arm*.

Abstract (≤ 150 words): Large video JEPAs are too slow for an embedded robot GPU (526 ms per two-view encode on a
Jetson Orin NX) and, zero-shot, predict our clips only 10–16 % better than copying the last frame. We distil a 326 M
V-JEPA 2 teacher into the 12.8 ms world model that runs on the robot, on 653 teleoperated SO-101 episodes, and compare
teacher, student, a random-teacher student and the same model trained from scratch on three probes: joint recovery,
ranking expert action chunks, and inverse dynamics beyond the arm's state. The student recovers the joints with R²
[to run] (scratch 0.565), ranks the expert chunk first [to run] (scratch 0.60) and adds [to run] beyond state
(scratch +0.007), at [to run] ms. [The teacher's knowledge does / does not survive the shrink at this scale.]

Sections: 1 Intro · 2 Teacher, student, rig, data (653 episodes, 222k frames) · 3 Distillation variants and the
random-teacher control · 4 Probes and bars · 5 Results: teacher → student → scratch · 6 Latency on the Orin · 7 Related
work · 8 Limitations (one rig, one teacher, fixed-step budget).

Figures: (1) teacher / students / random teacher / scratch / W2′ per probe on one axis with bootstrap CIs — [to run];
S3 scratch and W2′ points today from `paper-vs-hardware/posts/2026-10-29-paper-vs-hardware-lejepa/expected/gate0.json`
and `wm_train/results/2026-10-12_g1b_range_w1.json`; (2) latency vs probe score: W1 12.78 ms (G11-A), V-JEPA 2.1-G
526 ms (Gate B), VLA-JEPA check 1973 ms (§29 step 0), students [to run]; (3) V-JEPA 2.1 Table 11 replotted beside ours
(distilled vs scratch gain on their tasks vs ours) — from the PDF.
