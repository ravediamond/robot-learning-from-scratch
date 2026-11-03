# Feasibility — "What a 13 ms latent surprise can see": a receptive field by perturbation class on real robot data

WS-108, 2026-11-01. Memo only (template `journal/prompts/2026-10-31_paper_feasibility_TEMPLATE.md`; prompt
`journal/prompts/2026-11-01_paper_jepa-surprise-receptive-field.md`): nothing was run. Load status of every gate from
`journal/papers/jepa-world-model/AUDIT_W1_LOADS.md` (26 trained, 1 random = G11-A, 0 unknown); only `trained` rows are
cited as evidence. EXPERIMENTS rows are cited by their current line in `journal/EXPERIMENTS.md` (extract:
`runs/WS-108/rows.md`). The prompt's "Design detail" is refined in §3b; **no bar is set here** — the open protocol
choices are listed for the owner and frozen later in a PROTOCOL.md.

## 1. Claim in one sentence

On logged real SO-101 rollouts, the prediction error of three latent monitors (W1 L2 surprise, W2′ surprise, VLA-JEPA's
own world-model loss) separates perturbed from clean anchors (AUROC ≥ 0.8 at ≤ 1/20 false alarms on clean successes)
for the action-mismatch classes (wrong action, timing shift) at the lowest ladder step, and for none of the
scene-only classes (displacement, lighting, jitter) and not for the natural failures; the claim is false if any
scene-only class is "visible" under the frozen reading rule, or if the action classes are not.
(This is the memo's prediction from G0 and W1-A; the paper's result is the table, whichever way it falls.)

## 2. Evidence in hand

| gate (EXPERIMENTS line) | date | n | result | bar | verdict | pre-reg. | load |
|---|---|---|---|---|---|---|---|
| G0 (63) | 10-11 | 192 windows, 25 val episodes | K = 30 mean-error ratio vs true actions: cross-episode 3.34, first action held 2.54, zero 3.98, shift −15 / +15 steps (0.5 s) 1.99 / 1.58 (`wm_train/README.md` l. 168–174) | ratios ≥ 1.3 | **pass** | same-commit | trained |
| G11 Phase A (84) | 10-18 | timing loop | 12.78 ms mean, p95 12.87 ms per check | — | timing valid | — | **random** (G11-A, 85): latents void, timing stands |
| W1-A (98) | 10-19 | 50 (8 fail / 42 ok) | L2 surprise 2/8 at 2/42, AUROC 0.86; cosine drift 0/8 (0.30); per anchor: 23 % of 996 success anchors > 2.2 (`2026-10-19_w1_screening/README.md`) | ≥ 5/8 at ≤ 1/20 | **fail** | bar-first | trained |
| G11-B-cal (100) | 10-19 | 69 (25 fail) | 3/25 at 1/44 | recall ≥ 50 % | **fail** | bar-first | trained |
| W1-C1 (102) | 10-19 | 65 | imagined 0.70 < current 0.82 | ≥ 0.85, +0.05 | **fail** | bar-first | trained |
| W1-C2 (103) | 10-19 | 47 | successes 22.7 % flagged, failures 46 % | ≤ 1/20, ≥ 50 % | **fail** | bar-first | trained |
| REAP v2 (117) | 10-23 | 50 (8 fail) | logistic 3/8 at 5/42, AUROC 0.789 (shuffled actions 0.708, zero 0.693) | ≥ 5/8 at ≤ 1/20 | **fail**; action-conditioned (+0.07–0.08) | bar-first | trained |
| SURVEYOR gate 0 / 1 (121, 122) | 10-27 | 69 | τ 1.249 (paper 0.20), AUROC 0.71; monitor 3/8 at 6/42 (S = 30) | τ ≤ 0.40; ≥ 5/8 at ≤ 1/20 | **fail** | bar-first | trained |
| VLA-Corrector (ZJU) monitor (141) | 11-01 | 50 | 8/8 at 40/42; AUROC 0.72 vs shuffled actions 0.67 | ≥ 5/8 at ≤ 2/42 | **fail** | bar-first | (own model) |
| §29 steps 0–1 (191) | 10-25 | 50 timed checks; 10 trials replayed | VLA-JEPA loss runs on logged frames; p50 / p95 1973 / 1989 ms, VRAM 5.9 GB; failure ep 58 max 1.546 vs successes 1.505–1.555 (n = 1 failure, no claim) | measured | bar pending | — | trained (key set: 351 world-model keys, 0 missing, `jepa-world-model/gaps/vla_jepa_keyset.json`) |
| G9c (79) | 10-16 | 88 | HSV-mask judge, AUROC 0.919, 0 FP — the object masks for class 1 | ≥ 0.95 | fail (judge); masks usable | bar-first | — |
| Rollout labels | — | 69 (44 ok / 25 fail) | failure modes: wrong_object 9, missed_grasp 7 (+1 retried), stalled 6, blank 2; **slip_drop 0** (`data/rollouts/so101-bowls-policy-rollouts_labels.csv`) | — | — | — | — |

**Counts.** 11 gate rows: 1 pass (G0, the only perturbation-style evidence, a mean ratio, not per-anchor detection),
1 timing (G11-A, random encoder), 7 monitor fails, 1 measured (§29), 1 judge. Pre-registration: the seven monitor
gates bar-first, G0 same-commit. Blind labels: none. Statistic: W1-A and later use the episode bootstrap (2,000); G0
predates it (point ratios, recomputable from `wm_train/results/2026-10-11_mv_100ep_*.json`, $0). **No existing number
is a per-class detection rate**: the whole table of the paper is `[to run]`.

## 3. Gaps, costed

| # | gap | resource | queued? |
|---|---|---|---|
| 1 | Perturbation library `shared/perturb/` (7 classes × 3 steps, applied on the fly to decoded frames — disk is 8.2 GB free, floor 6 GB, so nothing perturbed is stored) | $0, ~0.5 Orin day (writing + unit checks per class) | new |
| 2 | Pixel scale for class 1: no px/cm or bowl size anywhere in the repo or `data/FACTS.json` | owner: one bowl diameter in cm (minutes), or ladder in bowl diameters | new (owner) |
| 3 | Class 6 footage: **0 slip_drop** among the 25 labelled failures (row above) and 0 in the other label files searched | owner: drop the class, or replace it with missed-grasp replay (8 episodes), or record slips (≈ 0.5 rig day) | new (owner) |
| 4 | M3 has no recorded-action input: its predictor is conditioned on Qwen action tokens computed from the frames (`2026-10-24_vla_jepa_surprise/README.md` l. 32), so classes 5 and 7 are undefined for M3 as written, and the sanity bar names M3 | owner: define M3's class 5 / 7 (e.g. action tokens from another episode's frames), or move M3's sanity check to another class | new (owner) |
| 5 | M1 scoring: 69 episodes × ~23 anchors (W1-A: 996 + 597 anchors) × 21 cells ≈ 33.5k checks at 12.78 ms ≈ 7 min compute; decode + perturb dominate | $0, ~2 Orin h | new |
| 6 | M2 scoring: W2′ checkpoint on the Orin (`~/robot-data/wm/runs/20261023T1054_train_57347595/out/w2/weights_final.pt`); **its Orin per-check time is missing** (the prompt's 1.8 s has no file) | $0, timing minutes, scoring ≤ 0.5 Orin day | new |
| 7 | M3 scoring: 50 × 10 anchors × 21 cells = 10,500 checks × 1.973 s ≈ 5.8 h; + clean 500 and natural 25 × 10 = 250 checks ≈ 0.4 h; at W1-A's ~23 anchors ≈ 14 h. Resumable | $0, ~0.5–1 Orin day | new |
| 8 | Natural class N: M1 clean scores exist (`outputs/w1_screening/latents`, `w1a_summary.json`); M2, M3 new (in 6, 7). Policy confound: 17 of 25 failures are TurboVLA variants, 2 of 44 successes are (labels file) → report N also within `smolvla_vlacpp*` (15 ok / 5 fail) | $0 | new |
| 9 | Trained-W1 live latency: re-time the check through `W1LiveMonitor.from_checkpoint` (strict, path C) — G11-A's 12.78 ms was a random encoder (same architecture) | $0, ~15 min | new |
| 10 | Training-overlap check: WS-091's G10 row says the 69 rollouts were "seen as unlabelled video in training"; if W1 saw the clean frames, clean surprise is deflated and every class inflated → verify from the training `split.json` | $0, minutes | new |
| 11 | PROTOCOL.md frozen with the owner's choices (3b) before any number | owner | new |
| **total** | | **$0 vast, ~2 Orin days, 0 arm days (≈ 0.5 rig day only if slips are recorded)**; no training (all three monitors exist) | |

### 3b. Design refinements (open protocol choices; the owner decides, nothing here is a bar)

| item in the prompt's design | what the files show | choice needed |
|---|---|---|
| "next 30 steps as W1-A" | W1-A: anchor every 25 frames, context = anchor + 2 previous frames, k ∈ {15, 20, 25} (W1-A README l. 18); G0 used K = 30 | horizon: W1-A's (reuses its clean scores) or G0's |
| "≤ 1/20 false alarms on clean successes (the W1-A operating point)" | W1-A's 1/20 is per **episode** (max over anchors, 2/42); the perturbation is per **anchor** | FA unit: per anchor (perturbation classes) and per episode (natural class) |
| class 4 frame drop vs class 7 timing | for an action-conditioned predictor, a stale observation with current actions is an observation/action misalignment, i.e. class 7 with a negative shift | class 4 as "last context frame repeated" (frozen camera) so it differs from 7 |
| sanity bar "class 5 random episode ≥ 0.9 AUROC for M1 and M3" | G0's 3.34 is a mean ratio over 192 windows, not per-anchor AUROC; W1-A's per-anchor surprise is heavy-tailed (p90 3.00 on successes at k = 3) | none on M1 (the bar stands as written); M3: gap 4 |
| class 6 slip replay | no slip footage | gap 3 |
| ~10 anchors for M3 | ~23 per episode at W1-A's spacing | anchor subsample rule for M3 (fixed, phase-stratified) |

## 4. Related work (all read from the mirror by PDF, except where marked)

| paper | monitor, data | per-class detection reported? | different here |
|---|---|---|---|
| ContactGuard 2608.13438 | probe on action-conditioned JEPA rollout, real Piper | **no**: per task, AUC 0.898–0.992 (Table 1); action corruption only as ablation (Table 3) | scene classes, three monitors, raw surprise not a trained probe |
| VLA-Corrector (Song) 2609.06508 | learned risk verifier, LIBERO-Plus + real | **no** for detection (AUROC 0.902 overall); *success* by perturbation (Table 3) | detection per class |
| VLA-Corrector (Pan) 2607.01804 | latent drift E_t, sim + real Piper | no; success by task group (Table 5) | failed here: 8/8 at 40/42 (row 141) |
| FAIL-Detect 2503.08558 | logpZO / RND + conformal band, sim + real Franka | **by scenario** (ID vs OOD, disturbance), ~78 % / ~72 % balanced acc. (Figs 4, 5); camera raised 10 cm, distractors | closest; ours is graded (3-step ladder) and per class |
| SAFE 2506.09937 | MLP/LSTM on VLA embeddings | no; per benchmark, unseen ROC-AUC 78.00 (Table 1) | — |
| ReAP 2610.16309 | visual + action-conditioned risk | no; bTWA 0.701 LIBERO-10 (Table 1a) | REAP v2 failed here (row 117) |
| FARM 2609.11445 | readout on frozen world-model state, incl. SO-101 | no; per task, pooled AUROC 85.68 (Table II(a)) | — |
| Spotter 2609.36808 | VLM screener | no | — |
| LeWM 2603.19312 | VoE surprise | **two classes** (colour change, teleport), qualitative only (Fig 10, sim) | the only JEPA surprise-by-perturbation result: sim, no number |
| VLA-JEPA 2602.10098 | not a monitor; policy success by LIBERO-Plus dimension (Table 3: camera 63.3 … light 95.6) | — | its predictor used as monitor M3 |
| SaiVLA-JEPA 2610.38508 | not a monitor | — | — |
| SURVEYOR (WS-050) | accept rule on W1 latents | — | failed here (rows 121–122) |
| VeriVLA 2610.21985 | not on disk, not read here | — | — |
| OOD detection for latent dynamics models | not on disk, not read here; the memo names none it has not read | — | to add to Drive |

**Which report per-perturbation detection:** none of the eleven read report detection AUROC per perturbation class on
a real robot; FAIL-Detect reports by scenario (two OOD settings), LeWM shows two classes qualitatively in simulation.

## 5. Objections

**(a) "Synthetic perturbations are not failures."** The claim is about the detector's receptive field; the natural class
(25 labelled failures, unperturbed) is a row of the same table — with its policy confound shown (gap 8).
**(b) "Clean frames were in W1's training set."** If gap 10 confirms it, the clean scores are deflated; answer: report
the episodes W1 never saw, if any, or state it as the condition; W2′'s encoder is frozen and M3 was trained on
teleop demos, not these rollouts.
**(c) "One world model / offline only."** Three monitors, two encoders, one policy predictor; offline stated; the live
cost is one re-measured number per monitor (gap 9).
**(d) "Negative again."** The table is the result; a visible action class and invisible scene classes is a positive
statement about what the monitor papers' signals can and cannot carry.

## 6. Venue

Primary: a 2027 workshop on failure detection / world models (RSS or CoRL workshop; deadlines to confirm when announced;
4 pages, double-blind). Fallback: TMLR (rolling). Overlap: WS-070 (`journal/papers/failure-monitors/`) and WS-091
(`journal/papers/jepa-world-model/`, monitoring uses) — this table could be a section of either. Recommendation:
**standalone 4-page**, citing both; it is the only one with a new measurement. arXiv first: after the table exists.
Endorsement (cs.RO): no contacts list in the repo; none named.

## 7. Minimal vs full paper

4 pages: M1 and M2 over the classes with source footage (1, 2, 3, 4, 5, 7) + N; the trained-W1 latency; the heat-map.
Needs gaps 1, 2, 5, 6, 8, 9, 10, 11 (~1 Orin day). 8 pages: adds M3 (gaps 4, 7), the slip class (gap 3), per-class
operating points for the seven failed monitors re-read as a map, and the within-policy natural class.

## 8. Risk of a null result

(1) Sanity fails (M1 class 5 < 0.9 AUROC per anchor): "nothing else is read" — the paper becomes a short note that
W1's raw L2 surprise does not see even an episode swap per anchor, which contradicts G0's mean ratio 3.34 and is itself
reportable as the "mean ratio ≠ detection" result. (2) Every class invisible: publishable as the receptive field
being empty at a 1/20 FA rate, explaining the seven failed monitors. (3) Scene classes visible: a positive result for
the monitor literature, and a reason to retest W1-A's failures by type.

## 9. Recommendation

**go-after-X**, X = the owner's four protocol choices (gaps 2, 3, 4 and the 3b table) frozen in a PROTOCOL.md. Cost:
$0, ~2 Orin days, 0 arm days. Earliest: the 4-page draft about two weeks after the go (offline only; one Orin);
mid-November 2026 at the soonest.

## 10. Skeleton

Titles: (1) *What a 13 ms latent surprise can see*; (2) *The receptive field of latent surprise on a $200 arm*;
(3) *Seven failed monitors, one map: perturbation-class detection for JEPA world models on real data*.

Abstract (≤ 150 words): Latent-surprise monitors assume a world model's prediction error sees failures. On a real
SO-101 task, seven such monitors failed their pre-registered bars. We ask instead what the surprise sees. We inject
seven perturbation classes, each at three magnitudes, into 50 logged rollouts — object displacement, lighting, camera
jitter, frozen frame, wrong action, slip replay, timing shift — and score three monitors: a LeWM-style JEPA trained on
180 demos (W1), a frozen-DINOv2 variant (W2′) and VLA-JEPA's own predictor. At 1/20 false alarms on clean anchors,
[to run] of 7 classes are visible at their smallest step; the 25 real failures reach AUROC [to run]. The trained W1
check costs [to run] ms on a Jetson Orin NX. The table predicts which failure-monitor claims can transfer to this arm.

Sections: 1 Intro (seven failed monitors) · 2 Rig, data, three monitors · 3 Perturbation library · 4 Protocol (bars
frozen before numbers) · 5 The table · 6 Natural failures and the policy confound · 7 Related work · 8 Limitations.

Figures: (1) 7 × 3 AUROC heat-map per monitor with row N, live cost in the header — `[to run]`; (2) per-anchor surprise
distributions, clean successes vs failures — `shared/gate/data/2026-10-19_w1_screening/` (today, W1 only); (3) G0's
corruption ratios vs per-anchor AUROC for the same corruptions — `wm_train/results/2026-10-11_mv_100ep_*.json` (today)
+ `[to run]`; (4) the seven failed monitors placed on the map — EXPERIMENTS rows 98–141 (today).
