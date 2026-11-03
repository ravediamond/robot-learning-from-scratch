# Feasibility — "A subtask token instead of joints": does language conditioning move a small JEPA latent off kinematics?

WS-109, 2026-11-01. Memo only (template `journal/prompts/2026-10-31_paper_feasibility_TEMPLATE.md`, prompt
`journal/prompts/2026-11-01_paper_jepa-language-conditioned.md`): nothing was run, trained, annotated or called. Every
W1 gate cited is `trained` in `journal/papers/jepa-world-model/AUDIT_W1_LOADS.md` (26 trained, 1 random = G11-A, not
cited). EXPERIMENTS rows are cited through the two sibling memos that carry their line numbers:
`journal/papers/jepa-world-model/FEASIBILITY.md` (WS-091, "W91") and `journal/papers/jepa-data-scale/FEASIBILITY.md`
(WS-106, "W106"). No bar is written or changed here; where the pre-written design leaves a bar undefined, it is listed
for the owner (§3 gap 7).

## 1. Claim in one sentence

With the G0 recipe on the 653 cubes-bowls episodes and the predictor conditioned on a subtask string instead of (C1) or
in addition to (C2) the joint actions, the encoder latent carries the current subtask (linear probe accuracy C1 − C3
shuffled-label ≥ 0.15, CI > 0, B2) without carrying more kinematics (R²(C1) ≤ R²(C0) + 0.05, B1), and one downstream
use improves over joints-conditioning (B3, CI > 0); the claim is false if B2 fails — the label adds capacity, not task
information — or if a probe from the commanded joints alone reads the subtask as well as C1's latent (§3 gap 5).

## 2. Evidence in hand

No gate has trained or probed a language-conditioned world model on this rig. What exists is the C0 side, the labels'
state and the controls that frame the question:

| gate (line, via) | date | n | result | bar | verdict | pre-reg. | role here |
|---|---|---|---|---|---|---|---|
| G0 (63, W106) | 10-11 | 25 val ep. | open loop 0.31 vs copy-last 0.89; 100 epochs, 12.1 h, $3.30 (+ $2.96 failed) | beats copy-last | pass | same-commit | recipe, cost unit |
| G1b W1 (65, W106) | 10-12 | 703 anchors | top-1 0.60; action-only floor 0.68–0.90 (69) | ≥ 0.8 | fail | bar-first | B3 probe (needs actions: C0, C2 only) |
| LeJEPA gate 0 + corr. (120, 123, W106) | 10-27/28 | 8,447 val frames | W1 mean R² 0.565, anisotropy 0.639; W2′ 0.867 / 0.301 | R² < 0.80 | met | bar-first | B1 probe |
| §30 (128, W106) | 10-26 | teleop val + 69 | (ii)−max +0.748; (v)−(iv) arm +0.007; random-init 1.254 vs W1 1.180 | three-part (`35f4600`) | not proven | same-commit | B3 probe |
| W1-A (98, W106) | 10-19 | 50 (8 fail) | 2/8 at 2/42 (AUROC 0.86) | ≥ 5/8 at ≤ 1/20 | fail | bar-first | probe (needs labels on the 50 rollouts for C1) |
| G10 step 1 (82, W106) | 10-17 | 69 | AUROC W1 1.000 | ≥ 0.90 | pass (retro.) | same-commit | must not fall |
| WS-091 gap 2 (`gaps/README.md`) | 11-01 | 69 (25 fail) | rest readout W1 0.9545 [0.902, 0.991] > W2′ 0.8655 > joints 0.7573 > policy id. 0.5882 | none (exploratory, in-sample) | — | no | W1 reads outcome beyond the joints |
| WS-091 gap 3 (same) | 11-01 | 65 scored (21 fail) | W1-C1 current latent AUROC W1 0.8199, W2′ 0.7067, joints 0.5305 | none (exploratory) | — | no | same |
| WS-062 A1 pilot 1 (`projects/system2-g05/README.md` §1) | 10-30 | 20 ep., 28 boundaries | Gemini 3.5 Flash-Lite: 1/28 within ±0.5 s (3.6 %); 1/12 without 0.0 s starts | ≥ 70 %; stop < 50 % | **stop** | bar-first | the labels |
| WS-062 A1 pilot 2 (same) | 10-30 | 20 ep., 34 boundaries | Gemini 3.5 Flash: 2/34 (5.9 %); "put" boundary ~0.6–0.8 s after gripper close | same | **stop** | bar-first | the labels |

**Counts.** 10 rows: 0 on a language-conditioned model; 6 W1 gates (C0's battery at S3, all `trained`); 2 exploratory
controls; 2 annotation stops. Pre-registration: the W1 rows keep WS-091's classes (3 bar-first, 3 same-commit); the two
A1 pilots are bar-first (bars in `journal/prompts/2026-10-30_system2_g05_subtasks.md` before the pilot). Blind labels:
none. Statistic: W1-A and the gap-2/3 controls use the episode bootstrap (2,000); G1b and G10 are point estimates,
recomputable from `projects/vla-jepa-world-model/wm_train/results/` ($0).

**Two facts that change the framing.**
1. **The premise is weaker than the item's `why`.** "The latent is the arm" (WS-091 §1) does not hold on two uses: at
   rest and at the grasp anchor W1 > W2′ > joints (`gaps/README.md`, "What this changes"). W1's latent already carries
   outcome information the joints do not, in-sample. The question becomes "does language add task (subtask)
   information the joint-conditioned latent lacks", not "does language rescue a latent that is only kinematics".
2. **LaLa-World is not this experiment.** Its "natural-language actions" are the numerical control actions verbalised
   with fixed per-domain templates, at step level (2610.27851 §3.1, Fig. 3, App. D.1); video conditions are captions
   by Qwen3-VL-30B-A3B. No subtask strings, no equal-data language-vs-joints comparison, no shuffled-label control, no
   probe of task information in the latent (all "not found", extract `~/scratch/WS-109/lala.txt`). Its Table E: on
   Push-T, numerical language actions 81.2 vs *qualitative* ones 4.2 — coarse wording destroyed planning. A subtask
   token is the coarse end of that axis. LaLa-World is related work, not prior evidence that C1 works.

**Labels: the gate the design itself sets.** The prompt: "if the annotation stops below 50 % boundary agreement, the
paper does not run". Both pilots stopped (3.6 %, 5.9 %). The annotator choice (Qwen3-VL-235B pilot, or stop) is open
in WS-062's `needs-from-owner`. Today, the paper does not run.

**Refinements to the pre-written design** (the memo refines; a PROTOCOL.md freezes; no bar changed):

1. **G1b is undefined for C1, C3, C4.** G1b ranks candidate action chunks through the predictor; a predictor with no
   action input cannot rank them. B3's "G1b top-1 … for C1 or C2" reduces to **C2 for G1b, C1 / C2 for §30**.
2. **§30's bar is three-part** (W106 correction 1, `35f4600`). B3 says "§30 Δ": which conjunct (or all three) is the
   owner's to state (gap 7).
3. **Probe on which latent.** R², §30, G10 step 1 and the subtask-id probe read the **encoder** latent. Conditioning
   the predictor on the subtask *gives* it the label, so the encoder is under less pressure to encode it: C1 can
   lower subtask decodability in the encoder. The protocol must name the probed tensor (encoder `z_t`, or predictor
   output `ẑ_{t+1}`); both are $0 to log.
4. **A joints floor for B2.** For pick / place tasks, subtask ≈ gripper phase (A1's own reference is joint 6 crossing
   mid-range). A linear probe from the commanded joints to the subtask id is the floor B2 needs, as the action-only
   floor was for G1b (W106 correction 2). Reported beside B2; whether it enters the pass rule is the owner's (gap 7).
5. **What C3 controls.** Permuting labels across episodes keeps each label sequence's timing (subtask order and
   duration correlate with time-in-episode), so C3 removes episode identity, not all information. A within-time-bin
   permutation would be stricter; the protocol states which permutation and seed.
6. **C0 sharing with WS-106.** C0 "= S4 of WS-106" only if S4 is the 653 alone; W106 correction 3 recommends S4 =
   S3-train ∪ cubes-bowls. Labels exist (or will) for the 653 only. Either C0–C4 train on the 653 alone (C0 is then its
   own training, +$3.30) or the 157 bowls train episodes are annotated too (≤ $1.5 pro rata of the ≤ $5, estimate).
7. **Held-out sets.** The bowls held-out set (25 val, `split.json`) has no subtask labels; the encoder-only probes need
   none, W1-A on the 50 rollouts and G1b for C2 need labels at test time (annotate them, or the owner's labels). The
   25-episode cubes-bowls held-out set must be an episode-level split carved before training (seed, per-task share in
   the protocol); 21 tasks (`projects/so101-multitask-smolvla/README.md` §1) over 25 episodes is ~1 per task.
8. **Text encoder.** With K ≤ 8 strings, a frozen text encoder is a fixed table of K vectors followed by the learned
   projection into the predictor's AdaLN conditioning (`embed_dim: 192`, `config/lewm_mv_so101.yaml`); the encoder
   choice changes little. Recommendation: SmolVLM2-500M's text tower (cached on the Orin, `hidden_size` 960 in its
   `config.json`), mean-pooled, frozen — the same tower SmolVLA reads instructions with. C3 uses the same vectors.
9. **Steps, not epochs.** G0's ≈ $3.30 is 100 epochs on ~53k train frames. The 653 are 222k frames
   (`journal/prompts/2026-10-30_system2_g05_subtasks.md`), ~3.6× S3's 61,462 (W106): fixed **steps** keeps ≈ $3.30 per
   training (~28 epochs); fixed epochs is ≈ $12 each. The prompt's "≈ $3 each" assumes fixed steps; the protocol says so.

## 3. Gaps, costed

| gap | what | resource | queued? |
|---|---|---|---|
| 1. **Labels that pass A1** | an annotator that clears WS-062 A1 (≥ 70 %; < 50 % = this paper does not run); then all 653 (+ 157 bowls train, + 50 rollouts if W1-A is kept) | owner's pick; Inference Providers ≤ $5 (WS-062 cap) + ≤ ~$1.5 (estimate, pro rata); 20-ep. pilot "cents" (267 calls) | WS-062 (waiting-owner) |
| 1′. $0 alternative | deterministic labels from gripper events (joint 6) — reach / grasp-carry / release; no push / flip subtasks | $0, ~0.5 Orin day; **but** labels are a function of the joints, so B2 measures kinematics by construction: only C4 (task string) stays non-kinematic | new (owner's call) |
| 2. Trainings | C1, C2, C3, C4 (C0 shared if gap 6 of §2 allows; else +1) at G0's step count | 4 × $3.30 = **$13.20** vast (+ $3.30 if C0 is separate); contingency per G0's retries ≈ $3 each | new |
| 3. Probe battery | R² (corrected protocol), §30 three-part, G10 step 1, subtask-id probe on encoder and predictor latents; G1b for C0 / C2; W1-A for C0 / C2 (C1 with rollout labels) | $0; ~8 Orin h per checkpoint (W106 gap 2) × 5 ≈ **40 Orin h** | new |
| 4. Shuffled-label control | C3, in gap 2 | in gap 2 | — |
| 5. Joints floor for B2 | linear probe commanded joints → subtask id, same split | $0, minutes | new |
| 6. Seeds | one extra seed for C1 and C3 (the B2 pair), so B2's CI is not one training's luck | 2 × $3.30 = $6.60 (optional) | new |
| 7. Protocol | bars B1–B4 copied from the prompt; owner states: §30 conjunct in B3, whether the joints floor (gap 5) enters B2, encoder vs predictor latent, permutation for C3, C0 on 653 vs union | owner ~1 h; PROTOCOL.md frozen before any training | new |
| **total** | 1 + 2 + 3 + 5 + 7 | **≈ $20 vast + annotation** ($13.20 + ≤ $6.5 labels; ≈ $30 with C0 separate, contingency and seeds), **~2 Orin days, 0 arm days** | |

## 4. Related work

| paper | what it showed | different here |
|---|---|---|
| LaLa-World 2610.27851 | templated verbalised control actions as the predictor's input; Qwen3.5-0.8B predictor; sim avg 72.1 vs JEPA-WMs 64.6 (Table 1); smallest set Rope 18,525 transitions (Table G); qualitative vs numerical language 4.2 vs 81.2 on Push-T (Table E) | subtask strings, not verbalised actions; one real arm, 653 episodes; a shuffled-label control and a latent probe, neither in LaLa-World |
| LeWorldModel 2603.19312 | the recipe (JEPA + SIGReg from pixels), action-conditioned | the conditioning signal is the only change |
| VLA-JEPA 2602.10098 | JEPA world model inside a VLA, language in the policy | its action head runs on our arm (W91 §4); here language conditions a standalone world model |
| SaiVLA-JEPA 2610.38508 | future-latent head +0.03 on control, 10,030 LIBERO-Plus episodes (W91 §4) | language-conditioned VLA with JEPA head; we probe the world model's latent, not control |
| JEPA-WAM 2608.09381 | latent world action model in a pretrained VLM backbone (INDEX one-liner) | on disk, not read for this memo |
| FineART 2609.36416 + LeRobot "Prompting Robots" Part 2 | 40,543 episodes, 533,913 subtask labels; G0.5 case study: subtasks did not change success, a demonstration habit beat `/subtask` (WS-062 item text) | the label source; **not on disk, not read here** |
| VLA-Corrector 2609.06508, RACE 2610.05719 | stage-aware state understanding; subskill-transition timing (INDEX one-liners) | subtask timing as a signal; on disk, not read here |
| RoboAtom 2610.40207, TS-WAM 2610.21922 | **not on disk, not read here** | to add to Drive before related work is written |
| latent-action papers | none found by title in `~/workspace/papers/INDEX.md` besides LaLa-World | — |

## 5. Objections

**(a) "Noisy labels."** Today the labels do not exist at a usable quality: 3.6 % and 5.9 % boundary agreement against a
70 % bar. Answer: the paper runs only after an annotator clears A1; the pilot rate is reported; C3 bounds what label
noise *alone* can do (C1 − C3), not what systematic lag (the steady ~0.6–0.8 s of pilot 2) can do — C4 (no timing) is
the check on timing.

**(b) "Language just adds capacity."** C3 has the same projection and the same vectors, labels permuted; C2 vs C0 has
the extra input; B2 is defined on C1 − C3. Residual: C3 keeps time-in-episode structure (§2 refinement 5).

**(c) "The subtask is in the arm's pose anyway."** New, and the strongest: for grasp / place, subtask ≈ gripper phase.
Answer: the joints → subtask-id floor (gap 5), reported beside B2; if C1's latent does not beat it, language added
nothing kinematics did not carry.

**(d) "Small data again."** Stated as the condition (653 episodes, 222k frames, 21 tasks). LaLa-World's smallest set is
18,525 transitions, ~12× fewer frames than ours, but in sim and with verbalised actions.

**(e) "Why not a VLA."** The latent is used for monitoring and verification (W1-A, G10, §30), not control.

**(f) "W1 is not only kinematics."** Answered by §2 fact 1: the paper's motivation is task information beyond the
joints, which W1 shows in part (gap 2/3 controls, in-sample); the test is whether language adds to it.

## 6. Venue

Primary: fold into the WS-106 curve paper as one section ("conditioning, not data") — C0 is shared with its S4 if
S4 is the 653 alone, and the probe battery is the same script. Standalone: a world-model / robot-learning workshop at
ICLR 2027 (4 pages, double-blind; dates to confirm on the workshop page). Fallback: TMLR. arXiv first: no, not before
the labels exist. Endorsement: no contacts list in the repo.

## 7. Minimal vs full paper

4 pages (or one section): C0, C1, C3 on the 653; B1, B2 with the joints floor; R² and the subtask-id probe only. Needs
gaps 1, 2 (three trainings, ~$10), 3 (encoder probes), 5. 8 pages: adds C2, C4, §30 / G10 / G1b / W1-A, the second
seed for C1 / C3, and the timing analysis (C4 vs C1, B4).

## 8. Risk of a null result

High on two fronts. (i) The labels: two annotators stopped at < 6 %; if a third does too, the paper does not run (the
design's own rule), unless the owner accepts gap 1′, which collapses B2 into kinematics. (ii) The result: LaLa-World's
Table E (coarse language 4.2 vs 81.2 numerical) predicts a subtask token conditions the predictor weakly, and §2
refinement 3 predicts C1 may move task information *out of* the encoder. A null with the shuffled control and the
joints floor ("at 653 episodes, a subtask token carries no task information the arm's pose does not") is a clean
paragraph in the WS-106 paper; standalone it is thin for a workshop.

## 9. Recommendation

**go-after-X, as a section of WS-106, not a standalone paper.** X = an annotator that clears WS-062 A1 (≥ 70 %; the
owner's pick, cents for the pilot). Then: protocol (gap 7), C1 / C3 (/ C2 / C4) at G0's step count. Total ≈ $20 vast
+ ≤ $6.5 annotation (≈ $30 with C0 separate, seeds and contingency), ~2 Orin days, 0 arm days. Earliest: the section
two weeks after the labels pass A1; with WS-106's curve, its submission date. If the owner does not pick an annotator
or the third pilot stops: **no-go** (the design's own rule).

## 10. Skeleton

Titles: (1) *A subtask token instead of joints*; (2) *Does language move a small JEPA world model off kinematics?*;
(3) *Conditioning, not data: language labels for a 653-episode world model*.

Abstract (≤ 150 words): A LeWM-style JEPA world model trained on ~180 SO-101 demonstrations reads the arm's pose (R²
0.565) and fails uses that need the task. We ask whether conditioning its predictor on a subtask string instead of the
joint actions changes what the latent carries. On 653 episodes of 21 tasks with subtask labels from [annotator, A1
agreement: to run], we train four variants (joints, subtask, both, shuffled subtasks) and an instruction-only control,
and probe each latent for the joints (R²), the current subtask (linear accuracy; chance 1/K) and three downstream uses.
The subtask-conditioned latent reads the subtask at [to run] vs [to run] for shuffled labels and [to run] for the
joints alone, with pose R² [to run]. [Verdict: to run.] Total cost [to run], 0 arm days.

Sections: 1 Intro · 2 Model, data, labels (A1 agreement) · 3 Five conditionings · 4 Probes and the joints floor · 5
Results · 6 Timing vs wording (C4) · 7 Related work (LaLa-World's verbalised actions vs subtask tokens) · 8 Limits.

Figures: (1) subtask-id accuracy C0–C4 with the joints floor and chance — [to run]; (2) pose R² per variant vs W1's
0.565 (`paper-vs-hardware/posts/2026-10-29-paper-vs-hardware-lejepa/expected/gate0.json`, today for C0 at S3); (3) A1
boundary lag histogram — `projects/system2-g05/results/a1_pilot_flash.json` (today, pilots), [to run] for the chosen
annotator.
