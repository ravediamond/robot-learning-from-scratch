# WS-082 · The gap map (step 0)

Inputs: `journal/EXPERIMENTS.md` (as of `7934eb4`), the site's `public/data/claims.json` (5 contradictions), the papers
library (`~/workspace/papers/index.json`, 141 files; WS-081's `CARDS.md` for the 99 unread papers). Reading only.
Excluded subjects (WS-069–074): reproducibility study, failure monitors / length null, edge-inference recipe,
self-improvement loop, length baseline, own-predictor monitor.

## (a) Not proven, parked, void and no-go rows (EXPERIMENTS.md)

| row | gate | status | reason |
|---|---|---|---|
| 41 | H0 (ensemble of chunks) | no-go | 0.76 overall vs null 0.92; 1.84× the time (643 vs 349 ms) |
| 47 | C1 probe (imagined vs current latent) | not proven | data confounded with policy: policy identity alone 0.76 beats every probe |
| 53 / 132 | BotFails (surprise on public SO-100 data) | parked | until G1 picks a winner (none so far) |
| 54–56 | RoboJEPA gates B, B′, A | no-go | 526 ms on the Orin; MPS fails; primary below its bar |
| 58 / 129 | G6a T3–T5 (done / stall detector) | stall dropped, T3–T5 pending | archive data; Jetson reachable |
| 67 | G11 phase B (live perturbation) | pending | phase A only |
| 68 | G11-A re-check | latents void | `strict=False` + key remap loaded a random ViT |
| 70 | G12 re-check | void | REKI on the Orin unmeasured (random-init model timed) |
| 84 | G15-v3 fit | θ_c pending | — |
| 95 → 100 | REAP v1 → v2 | v1 not proven, superseded by v2 fail | no monitor at the bar |
| 107 | HWM crossover gate 0 | not proven (owner) | open confound: ~0.6 h of one task vs the paper's data |
| 108 / 139 | PRIME | void | the graph check failed; no rollout number read |
| 111 | §30 inverse dynamics | not proven | random-init control blocks (not sample size) |
| 117 | Horizon × depth A1 | not proven | (A2 latency passes; gate B waits for `go arm`) |
| 127 | G9a (own) full | waiting | per-episode files on the old Mac |
| 128 | G9b (Robometer + LoRA) | not running | preprocessing fix, then budget go |
| 130 | G7a / G7b / G8 (world-model chunk selection on the arm) | waiting | cube-primitive session, W2 weights, onsets, G7 amendment |
| 131 | G3 (proposal prior) | waiting | trained policies on the new task |
| 133 | G15 stage tracker | fail; v2 fails its dev fit | owner's grasp / place labels |
| 134 | G13b (RTC guidance vs prefix conditioning) | waiting | SmolVLA checkpoint on a GPU |
| 135–136 | G16 repair vs retry, G17 edited stage program | waiting | judge τ, calibration, robot days |
| 164–170 | NEXT_ANGLES §24, §26, §29 | measured, no verdict / bar pending | blank validation lists, step 2 bars, vast fine-tunes |
| 169 / 172 / 174 | TurboVLA horizon | no claim, closed | instrument check failed twice |

## (b) The five register contradictions and the condition keys that differ

Each pair is a paper claim the register tested against the rig's own measurement. Within the register both sides
carry the rig's conditions, so the key that differs is mostly between the paper's setup (`theirs`) and the rig's.

| metric | paper claim | rig side | register keys that differ | what differs from the paper's setup |
|---|---|---|---|---|
| G1b top-1 | LeWorldModel: latent ranking > action-only (Push-T planning 90 %) | action-only 0.68–0.90 > W1 0.60 | none | Push-T sim, planning by CEM → SO-101 recorded candidates, ranking only |
| G1b top-1 | DINO-WM: latent ranking > action-only (Push-T 0.90) | action-only > W2 0.61 | `model` (W2 frozen DINOv2-S vs W1) | as above; the rig side is LeWM's row |
| failure AUROC | ContactGuard: imagined-latent probe > current (0.98 vs 0.84, Cup) | current 0.82 > imagined 0.70 | none | four grasp tasks, one policy → mixed-policy rollouts, 2 views (like-for-like) |
| failure AUROC | Robometer: learned reward model > hand-made signals | episode length 0.938 > Robometer 0.818 | none | the paper ran no length baseline (premise, not a table) |
| control-step delay | PACT-VLA: no added delay | 37 ticks vs 8 | none | their GPU vs Orin NX MAXN, vla.cpp sequential (like-for-like on the claim) |

Two are like-for-like (ContactGuard, PACT); three set the paper against a baseline it never ran.

## (c) What the clusters assume, and what that leaves measurable here

**World models** (LeWM 2603.19312, DINO-WM 2411.04983, V-JEPA 2-AC 2506.09985, Fast LeWM 2606.26217, DREAM-Chunk
2606.18589). They plan or rank with hundreds of imagined rollouts on a GPU nothing else is using, in scenes where the
camera sees the whole workspace (Push-T, a clear tabletop, several cameras) and the arm can reach every commanded pose.
Here the world model shares a 16 GB GPU with the policy, the top camera is oblique and the arm hides the bowl during
transport, and the SO-101 saturates at its reach and joint limits. Measurable here and nowhere in the cluster: where a
latent world model's error goes when the view is occluded by the arm itself, and whether it predicts physically
impossible motion past the arm's limits.

**VLAs** (π0.7 2604.15483, MolmoAct2 2605.02881, SO-101 benchmark 2606.08881, LoRA-on-SO-101 2512.11921,
Xiaomi-Robotics-1 2607.15330). Fleets of demonstrations, H100 serving, or per-task fine-tunes with 50–200 demos
evaluated by staff who reset scenes. None reports what the arm physically did against what the policy commanded:
on a Franka or YAM the tracking error is negligible. Here the $200 arm's servos have finite resolution, deadband and
load-dependent lag. Measurable here: how much of a VLA's error is the arm not following, not the policy choosing wrong.

**Monitors / judges** (Robometer 2603.02115, TOPReward 2602.19313, FailSafe 2510.01642, Spotter 2609.36808,
GVL 2411.04549). Human-annotated progress or curated success/failure datasets, judged offline. This cluster overlaps
the excluded failure-monitors and length subjects; no candidate is drawn from it here.

**Chunking / latency** (RTC 2506.07339, RACE 2610.05719, Real-Time EXPO-FT 2609.18207, PACT 2610.28019, AAC
2604.04161, smooth MP chunking 2610.39451). Parallel sampling and verification on an idle RTX 4090/5090 or H100, where
N candidates cost one batch; arms whose joints follow commands to well below the size of a chunk seam. Here every
extra sample is sequential on the Orin and is paid in delay, and the smoothing the smooth-chunking gate measured acts at
a fraction of a degree: projection (S2) moves the first action by 0.30–0.32° and the error vs recorded by −0.02°
while removing ~94 % of within-chunk jerk (README results table), a few encoder steps of the arm's servos.
Measurable here: the delay cost of test-time sampling at fixed wall-clock, and whether seam corrections reach the
joints at all.

**Memory** (HyMeS 2608.09410, Workspace Models 2609.20820, RoboTTT 2607.15275, StreamPI 2608.26067). Memory tasks are
built by hiding a cue on purpose (a removed marker, a count); context is long because the GPU can carry it. Here
memory is forced by the rig: the arm occludes the target bowl in the only top view during transport, every episode.
Measurable here: whether a short frame history repairs a natural occlusion, the case the cluster never had to stage.

**Instruction following** (CoVer-VLA 2602.12281, π0.7, VLAs-as-Tools 2605.13119, StreamPI, RoboHiMan 2510.13149).
Many instructions over many scenes, and language perturbations evaluated in simulation at scale. Here two
instructions share one scene and ~90 demos each: the layout alone can predict the task. Measurable here: at what
point a small VLA reads the instruction at all, and what wins when layout and instruction disagree.

**Data / teleop efficiency** (Mulligan 2610.05882, Ego2Robot 2608.02580, Robot Data Factory 2609.16705, FACTR 2
2606.12406, ACT 2304.13705). A lab operator or fleet collects data; operator time is not the reported cost. Here
teleop is the binding constraint (~2 arm days a week, 180 demos), and a public pool of 203 community SO-101 datasets
(50 h; 71 datasets / 14.4 h that match the sensors) exists with mismatched cameras and calibration. Measurable here: the
success value of one teleop hour, and how many own demos the community pool replaces.

**RL / self-improvement** (EXPO-FT 2605.25477, HiL-ResRL 2606.22860, SARM2 2606.10305, ENPIRE 2606.19980, SEAR
2610.18792). Human resets, and training on failed rollouts. The rig forbids training on failures and has no human
during rollouts; the excluded self-improvement-loop memo covers what is left, so no candidate is drawn here.
