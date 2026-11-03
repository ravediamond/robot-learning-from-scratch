# Chapter 12b. Planning with JEPA world models, and grounding them: the papers in depth

Chapter 12a covered the models. This chapter covers the two questions that decide whether a model is useful on a robot: **can the latent plan** (how far, with what cost, against what baseline), and **what the latent must contain** for the answer to be yes (physical state, decision structure, invariance to distractors). The eight papers here are the ones that asked those questions explicitly. Three of them are the papers behind your own gates: Planning Limits behind G1b, D-JEPA behind the proxy-label version in NEXT_ANGLES §1, and PSG-JEPA behind §7.

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | JEPA-WM study (Terver et al.) | 2512.24497 | The design-choice ablation of the whole family; the protocol everyone else inherits |
| 2 | Planning Limits of Latent World Models | 2609.39235 | Plannable range = rollout length; 92 → 41 % with a *perfect* predictor; VLA+WM 65 → 77 % |
| 3 | D-JEPA | 2609.24749 | The decision-local gap; a set operator on executed outcomes; real PiPER +17 pts |
| 4 | PSG-JEPA | 2608.06799 | Ground latents in proprio and multi-horizon Δq; real ALOHA 60 → 79 % |
| 5 | SA+IDM (Genisom) | 2609.03565 | State alignment beats IDM alone; straightening is not quality |
| 6 | CF-JEPA | 2610.00727 | Controllable/uncontrollable split; the only model that survives distractors |
| 7 | Imagine-RL | 2609.24033 | A LeWM-style visual–torque model as a critic feature for VLA RL; real PiPER 50 trials/task |
| 8 | FOREWARN | 2502.01828 | Decouple foresight (world model) from forethought (VLM); 10 trials/task |

---

## 12b.1 The JEPA-WM design study (Terver, Yang, Ponce, Bardes, LeCun; 2512.24497)

### What it is

Not a new model: a controlled ablation over the family "frozen visual encoder + learned action-conditioned predictor + CEM in latent space", with the stated goal of finding the best configuration. The authors release code (`jepa-wms`) and the baselines (DINO-WM, V-JEPA-2-AC) run under the same protocol. Read it as the field's own answer to "what matters", and as the protocol to copy when you compare two world-model recipes.

### Formalisation (their eqs. 1–5)

Training: L = (1/B) Σ L[P_θ(E(o_{t−w:t}), A_θ(a_{t−w:t})), E(o_{t+1})], MSE, frame-causal mask so one model is trained at every context length 0…W−1 simultaneously; frameskip f stacks f actions into one (f·A)-dim input. Multistep rollout loss L_k (eq. 5): the same loss on a k-step autoregressive unroll with truncated BPTT (gradient only through the last prediction). Planning (eq. 2): minimise L_vis + α L_prop between the unrolled terminal latent and the encoded goal; α = 0.1 when proprio is available, 0 on DROID/RoboCasa to match V-JEPA-2-AC.

### Protocol (§5.1), worth memorising

- Environments: Metaworld Reach / Reach-Wall (data from a TD-MPC2 agent), Push-T, Wall, PointMaze (DINO-WM's datasets, 90 % train), DROID → zero-shot RoboCasa Place / Reach (no fine-tune), plus 16 real Franka videos.
- Metric: success rate; on DROID, an **Action Score** = rescaled negative L1 between planner actions and the dataset's ground-truth actions (an offline proxy, like H-JEPA's Fréchet fidelity).
- Variance: **3 training seeds per final model; e = 96 evaluation episodes per epoch (64 on DROID, 32 on RoboCasa because each episode replans 12 times); score = mean over the last n = 10 epochs (100 on DROID)**; error bars = std across those epochs. They say openly "we still find high variability throughout training", which is why they average epochs instead of picking the best one.
- Design-choice order: planner first (fixed per environment thereafter), then training/architecture, then scale. "Each component is independently varied from the base configuration."

### Findings, each with its mechanism

- **Planner** (Fig. 3 left): CEM L2 best or competitive everywhere; gradient-based (Adam/GD) wins on Metaworld (smooth, greedy cost landscape) and fails on Push-T/Maze/Wall and DROID (local minima, multimodal). Nevergrad (NGOpt) matches CEM on real data with fewer hyper-parameters. L2 > L1 always.
- **Multistep loss** (Fig. 3 right): 1 → 2 steps helps; more hurts in sim (they plan with W^p = 2, so a 6-step loss trains for a task not performed at test); on DROID the optimum is 6 — real dynamics need more.
- **Proprio** (Fig. 4 left): consistently better; on Metaworld most failures are "reach quickly, then oscillate around the goal", which precise distance fixes. Not shown on RoboCasa because proprio spaces differ across DROID and RoboCasa.
- **Encoder** (Fig. 4 right): DINOv2/v3 > V-JEPA/V-JEPA 2 for these tasks ("better fine-grained segmentation"); DINOv3 wins only on photorealistic DROID/RoboCasa and converges slower on Maze/Wall.
- **Context W** (Fig. 5 left): W = 1 → 2 is the big jump (velocity needs two frames); longer context trades fewer gradient steps for more context; DROID optimum W = 5 vs 3 in sim. Hard rule: plan context W^p ≤ training W, or predictions degrade rapidly.
- **Architecture** (Fig. 6 left): AdaLN + RoPE best on average, but task-dependent and the margin is slight; feature-conditioning (DINO-WM style) wins on Metaworld.
- **Scale** (Figs. 5 right, 6 right): no gain from ViT-S → L or depth 3 → 12 in simulation (and larger latents make nearby states harder for CEM to distinguish, S5.9); clear positive correlation on DROID. Optimal depth ≈ 6 in sim, possibly 3 on Wall/Maze.
- Final models (Table 1, mean (std)): Push-T 70.2 (2.8) vs DINO-WM 66.0 (4.7); DROID 48.2 (1.8) vs V-JEPA-2-AC 42.9 (2.5); RoboCasa Place 30.7 (8.0) vs V-JEPA-2-AC 33.1 (7.2) — one loss honestly bolded for the baseline.

### What is not there

- No real-robot closed loop (the 16 Franka videos are offline).
- No trivial-controller baseline.
- "Even with models which are able to faithfully unroll a large number of actions, success at the planning task is not an immediate consequence" (§5.2, E.1) — stated, with visualisations, but not quantified; 2610.19375 later quantifies it as r ≈ −0.01.

### For the SO-101

Four rules transfer as-is to W1/W2′ training: context ≥ 2 frames; a 2-step rollout loss; proprio in, with α small in the cost; plan context ≤ train context. The evaluation habit to copy is "mean over the last n epochs, three seeds" instead of "best checkpoint" — it is the cheap defence against the checkpoint-selection problem that 2610.19375 shows is real.

---

## 12b.2 The Planning Limits of Latent World Models (Alrasheed, Azam, Akhtar; 2609.39235)

### The question

Not "what can the model do" but "how far ahead can it rank actions, and what sets that distance". The paper's own vocabulary: **rollout horizon K** (steps the planner imagines) vs **target lookahead L** (steps to the goal); the **plannable range P\*** = largest L at which the model still ranks expert actions well.

### Protocol (§3)

- Five frozen ViT-L encoders (V-JEPA 2, V-JEPA 2.1, VideoMAEv2, VideoPrism, DINOv2), same predictor architecture, tuned per encoder; none adapted to Meta-World.
- Meta-World 16 main tasks (13 train / 3 held out), 4 articulated-object diagnostic tasks, 4 near variants, 4 novel tasks; **2 958 BridgeData V2 WidowX episodes** for the offline real-data check.
- **Ranking test**: at a point in a successful expert demo, compare the next expert sequence against **100 random alternatives**; score J_L(a) = min_{k≤K} [1 − cos(ẑ_{t+k}(a), z^{demo}_{t+L})] (eq. 1); report the **expert percentile** (fraction of random sequences scoring better; chance 50; threshold 10). K = 5, L ∈ {1,2,3,5,10,20}.
- Closed loop (§6): V-JEPA 2.1, 16 tasks, **20 paired episodes per task**; pure imagination vs MPC; goal image vs subgoals 20 / 5 steps ahead from an expert rollout of the same episode.
- VLA+WM (§7): fine-tuned π0 proposes N candidates; WM scores them against a subgoal; N swept 1–32.
- The decisive control: **replace the world model with the real simulator** and repeat the closed-loop test.

### Results

- Plannable range with K = 5 is **5–10 steps** for every encoder (Fig. 3c); tasks take 16–53 steps (Fig. 2b). On BridgeData the released V-JEPA 2-AC favours the recorded action one step ahead and is at chance ten steps ahead (§4.2).
- **An 81× larger predictor does not extend the range** (§5.1); recursive training from 1 → 5 steps extends it from 5 → 10, 20 adds nothing (Fig. 3a); only a longer rollout K extends it, to ≥ 20 (Fig. 3b).
- **The real simulator falls from 92 % to 41 %** when the target moves from 5 to 20 steps ahead of a 5-step rollout (Fig. 1b); the learned model reaches 40 % at 20 steps, "close to the simulator". So the limit is in the **score** (closeness to a distant target, which detours and similar end-distances defeat), not the predictor; Euclidean instead of cosine does not help (§5.5).
- Closed loop (Fig. 4a): pure imagination 23 % → 36 % with a 30-step rollout; MPC with a 5-step rollout 30 %; MPC + 30-step 52 %; **MPC + 5-step subgoals 76 %** (146 paired gains, no losses). Random-init predictor with the same subgoals: 0.8 % — the dynamics are needed, not just the subgoals.
- VLA+WM (Fig. 6): π0 65 → 77 % on 16 tasks, no task worse; untrained predictor as critic 3.6 %, WM 47.9 %, simulator 53.6 % on the 7-task subset — the learned WM gets 82 % of the oracle's gain; N = 16 is enough; near variants 25 → 49 %; novel objects ≈ 0 for both ("it cannot fix a policy whose proposals are all poor").

### What is not there

- Real robot in closed loop: none; BridgeData is offline ranking only.
- Error bars on the closed-loop numbers are paired counts (gains vs losses), not intervals; fine, but note the n = 20 per task.
- No proprio-only or commanded-action ranking baseline in the ranking test; the 100 alternatives are random sequences, which is the *easiest* negative set (they say so and move to perturbed-expert negatives in App. F).

### For the SO-101

This is the paper that explains G1b, and it does it with a control you could not run: the real simulator. G1b scored 7-step chunks against a goal many chunks away with terminal latent distance — exactly the K ≪ L regime where even a perfect model is at 41 %. The commanded-joints baseline winning (0.68–0.90 vs 0.60) is the same fact seen from the other side: when the target is far, "closeness" is a bad score and the thing that *is* monotone (the planned joint path) wins. Two things to carry: (1) if W1 is ever used to rank again, the target must be a subgoal within the rollout (a frame from a demonstration of the same layout, ~0.25 s ahead), and the random-sequence negative set is too easy — use perturbed demos; (2) the VLA+WM result is the shape of a test you can afford: SmolVLA proposes N = 8 chunks (one forward each, 238 ms × 8 off the loop), W1 scores against a near subgoal, paired episodes, bar on pooled success. It is the only "world model improves a policy" protocol in the family with no-task-worse evidence, and it needs no planner.

---

## 12b.3 D-JEPA (Liu, Wu, Wen, You, Zhang, Hao, Lin, Su; 2609.24749)

### The diagnosis (§3)

The "decision-local prediction gap": global predictive geometry is informative, but among the **few candidates that compete for execution**, latent goal distance misorders successes and failures. On a 96-start Push-T audit, LeWM prefers a failing candidate over an available success on 8/96 starts; pair-inversion rate in the top-4 shortlist is **49.0 % for LeWM, 38.2 % for TD-JEPA** (Fig. 2c); Spearman between predicted and realised cost within a start falls from 0.90 (all 63 candidates) to **0.11 (top 4)** (Fig. 2d). Proposition 1: global agreement cannot guarantee correct selection.

### Method (§4) — this is what NEXT_ANGLES §1 got wrong

D-JEPA is **not a loss added to the world model**. It is a separate **permutation-equivariant set operator** that takes the complete candidate set and outputs a bounded correction to each score. Per candidate i: descriptor d_i = LN(ẑ_{i,H} − z_g) and ordinal rank r_i from the native cost, from two models (LeWM, TD-JEPA): token v_i = [d_i^L; d_i^T; r_i^L; r_i^T], base score b_i = α r_i^L + (1−α) r_i^T. A shared encoder + 2 Set-Transformer layers (4 heads, 64-d) + rank-8 head give δ_i = ε tanh(W_up tanh(W_down h_i)), ε = 0.2, s_i = b_i + δ_i (eq. 1). Because |δ| ≤ ε, preferences with base gaps > 2ε are preserved; the operator can only reorder within 2ε of the base minimum (Prop. 2). Loss (eq. 2): L_R = −log Σ_{i: y_i=1} p_i + λ_local L_local + λ_trust K⁻¹ Σ δ_i², with p_i = softmax(−s_i/T) — success mass globally, local success/failure ordering near the boundary, and a trust penalty on unnecessary correction.

Supervision: **counterfactual execution** — every candidate in the set is executed from the restored start and labelled. That is the data cost, and it is why your proxy-label version (labels from a heuristic, not execution) is not this test.

Two complements: *predictive plasticity* (adapt only the last TD-JEPA predictor block + projection under the same objective; calibrated composition picks it over the relational default when its score margin clears a threshold), and *representation lifting* (write the learned rank back into the terminal latent, eq. 4: z̃ = z_g + π_i/(K+1) · u_i, so native latent distance recovers the aligned choice at 34 ms vs 52 ms, Table 6).

### Protocol and results

- Push-T confirmation cohort **n = 256 starts**, Reacher 128, Granular 64, candidates shared within tasks; **fitting / calibration / evaluation identities disjoint**; Wilson 95 % intervals in Fig. 10c; paired counts and bootstrap in App. C.
- Table 1: Push-T LeWM 83.59 → D-JEPA 87.89; Reacher 86.72 → 93.75; the "matched adaptations of published objectives" rows (PEGrad, Var-JEPA, conflict-safe, expected-value, CVaR planning) all 78–82 — the baselines are other people's methods re-implemented on the same candidates.
- Table 2, embodiment transfer: RoboTwin VLA 61.72 → 76.76 (n = 128 × 4 tasks, +15.04); **physical PiPER with V-JEPA 2-AC planner, n = 50 per task, paired starts**: Real Push-T 56 → 74, Two-cube stack 72 → 88 (+17 overall; gains/losses 12/3 and 10/2).
- Table 4: held-out PushObj shapes +9 over calibrated fusion; seven appearance shifts (blur, noise, dark, colour changes) +10.29 mean, every condition positive (50 fresh starts each).
- Table 5 ablation: plasticity alone +2.74; relational alone +10.16; both +10.94; dual-source descriptors 93.75 vs 84.38 single.

### What is not there

- The number of counterfactual executions per start (63 candidates × 256 starts in sim) is the hidden cost; on the physical robot the candidate pool and search budget are "retained from the reference planner", and the supervision for the real-robot operator is in App. A.5 — read it before claiming the real-robot result is label-cheap.
- Oracle (candidate-outcome ceiling) is reported only for driving (100.00).

### For the SO-101

The decision-local gap is your G1b in their words: W1's distance ranked chunks at 0.60 while the commanded joints ranked them at 0.68–0.90, and the set that matters is the few chunks a policy would actually propose. **W1-R** (2026-10-22) already ran the operator on frozen W1 with proxy labels (executed chunk = success, decoys = failure): regret 0.2631 vs W1's 0.2630, gate "never replace" in 24/25 seeds, and its README says plainly that it does not test D-JEPA's claim because there are no executed outcomes. The version that does: during the G7 cube session, execute K = 4–8 candidates from the same restored layout for a subset of starts, label by G9c-3 at rest, fit the same operator on those labels with disjoint fit / calibration / evaluation starts (NEXT_ANGLES_AMENDMENTS item 1). Bounded correction (ε) is also the right safety design for any re-ranker on the arm: it cannot overturn a confident native choice.

---

## 12b.4 PSG-JEPA (Yan, Zhu et al., HKUST-GZ; 2608.06799)

### The diagnosis

Forward-only LeWM latents do not reliably expose robot-centric state. Single-latent linear probes on OGBench-Cube (Table 1): joint position r = 0.71, end-effector position 0.99, gripper 0.93, **EE-yaw 0.08** (DINOv2 0.51). Pairwise probes (Table 2): gripper velocity linear r = 0.44.

### Method (§3), the whole thing in three equations

Keep LeWM (C = 3 context frames, single-step prediction, SIGReg λ = 0.09). Add two training-only heads on the shared latent:

- Static: L_static = (1/T) Σ MSE(H_s(z_{t+i}), s_{t+i}) — every latent regresses the proprio vector (joints, gripper, EE pose). (eq. 4)
- Dynamic: for every horizon k ∈ {1,…,T−1} and every valid pair, L_dynamic = (1/|K|) Σ_k (1/|I_k|) Σ_i MSE(H_Δ(z_{t+i}, z_{t+i+k}), Δq_{t+i,k}), Δq = q_{t+i+k} − q_{t+i}: **joint-angle change between endpoints**, not the action sequence (which is ill-posed: variable length, many sequences map to the same endpoints). Horizons weighted equally so short ones do not dominate. (eqs. 5–6)
- L_PSG = L_JEPA + λ_g (L_static + L_dynamic), λ_g = 0.1 fixed across all benchmarks. Heads discarded at inference. (eq. 7)

Baselines: LeWM; **LeWM_ActionIDM** (one-step inverse dynamics on adjacent pairs — the alternative everyone else uses, built by the authors as a matched control); frozen DINOv2.

### Protocol

Three levels, same encoder everywhere:

- **Q1 probes**: episode-level train/test split; linear ridge and shallow MLP; Pearson r. PSG: joints 0.83, EE-yaw 0.94/0.98, gripper velocity 0.69/0.76, action 0.80/0.86.
- **Q2 planning**: freeze encoder, train the same GC-IDM planner (3-layer MLP 512, AdaLN-zero) at 5/10/25/100 epochs and 100/25 % data; **200 goal instances; mean ± std over 3 planner seeds**; OGBench-Cube and -Scene. Table 3: at 5 epochs full data PSG 95.0 ± 0.7 vs LeWM 80.7 ± 1.9; at 25 % data, 100 epochs, 93.2 vs 83.5; the biggest margins at small budgets (Fig. 3). Open-loop latent MSE at 30 model steps 0.0485 vs 0.1488 (−67 %).
- **Q3 policy**: LIBERO-Goal, OFT action head, encoder fine-tuned jointly, 30 epochs, **3 seeds × 10 tasks × 50 rollouts = 1 500**: PSG 85.3 ± 3.9, LeWM 77.7 ± 0.5, ActionIDM 82.6, DINOv2 80.1. **Real dual-arm Cobot (Mobile ALOHA design, 3 cameras), 100 demos/task, 50 trials/task**: Place-to-Bread 84 vs 62, Place-to-Plate 74 vs 58, Pour-Water 80 vs 60.
- **Q4 ablation** (Table 5): removing transition grounding costs planning most (95.0 → 81.3); removing state grounding costs state-probe r most (0.94 → 0.69); multi-horizon vs adjacent/endpoint only: 95.0 vs 93.5/93.7 (within noise on planning, larger on policy 85.3 vs 81.5).

### What is not there

- Real-robot intervals: three tasks × 50 trials, point estimates only; no seeds on the real robot.
- 2610.19375's control (does a grounding head beat simply training LeWM longer?) is not run; the open-loop-MSE improvement is suggestive but is the proxy that paper shows is uncorrelated with planning.
- Which parts of the proprio vector matter is not ablated (joints vs EE pose vs gripper).

### For the SO-101

This is §7 of NEXT_ANGLES, and the reading changes the bar. Your arm supplies joints and gripper but no EE pose unless you compute FK; the paper's largest single-latent gain was on EE-yaw, which your loop task barely varies. The honest expected gain is on the *transition* side (Δq across horizons), which is cheap to add to W1 and directly tests whether the latent carries "where the arm went", the quantity G1b's commanded-joint baseline already has for free. So the pre-registered comparison for §7 must be three-armed: LeWM, LeWM+ActionIDM (the matched control they built), PSG, all with the 2610.19375 matched-epochs control, judged on the G7 object-state probe — not on open-loop MSE.

---

## 12b.5 SA+IDM (Liu, Huang, Liang, Gao; Genisom AI; 2609.03565)

A short paper that sharpens one point of 12b.4. Same LeWM backbone (ViT-Tiny/14, 192-d, 6-layer causal predictor, AdaLN, 3-step context). Two heads: inverse dynamics H(z_t, z_{t+1}) → a_t (eq. 4) and **state alignment** G(z_{t−1}, z_t) → s_t on a consecutive pair "because it provides the temporal context needed to align velocity measurements" (eq. 5). L = L_pred + α L_sa + β L_idm; α = β = 1.0 chosen on a Push-T validation split from {0.01, 0.1, 1.0}.

Protocol: LeWM's four-task benchmark, same 50 fixed start–goal problems, goal 25 steps ahead, 50-step budget, **3 independently trained seeds, baseline numbers taken from the LeWM paper** (so not strictly matched). Table I: TwoRoom 100 / Reacher 85 / Push-T 98 / OGBench-Cube 87 vs IDM-only 94 / 63 / 83 / 85 and LeWM 87 / 86 / 96 / 74. State alignment helps over IDM everywhere; LeWM still wins Reacher.

The second contribution is a warning about a metric: **temporal straightening** (cosine between consecutive latent displacements, eq. 8), often read as "good latent". LeWM has the highest straightening (0.69) and the lowest Cube success; SA+IDM the lowest (0.55) and the highest. The transition-subspace analysis explains it: r_95 (components holding 95 % of transition energy) is **17 for LeWM vs 32.8 for SA+IDM** — LeWM's displacements live in a low-dimensional subspace, so successive ones are mostly parallel or antiparallel, which inflates the mean cosine. "Average straightening alone can conceal the concentration of temporal variation in a low-dimensional subspace."

What is not there: intervals on Table I; a matched re-run of the baselines; any real robot.

For the SO-101: the r_95 measurement is a two-line addition to the effective-rank exercise in chapter 12.7 and should be run on W1 and W2′ side by side; if W1's transitions are low-rank, that is a candidate mechanism for G6 (progress by distance failed) that costs nothing to check.

---

## 12b.6 CF-JEPA (Byrd, Wright, Ha; Georgia Tech; 2610.00727)

### Method

Split the LeWM latent z ∈ ℝ^192 into controllable z^c ∈ ℝ^128 and uncontrollable z^u ∈ ℝ^64. Two forward predictors: g(z^c, a) → ẑ^c (AdaLN-zero on the action), h(z^u) → ẑ^u (no action); L_fwd on the concatenation (eq. 3). Inverse dynamics ψ(z^c_t, z^c_{t+1}) → â (eq. 4) **only on z^c**; **adversarial** inverse dynamics φ(z^u_t, z^u_{t+1}) → â through a gradient-reversal layer (eq. 5), so z^u is pushed to be *un*predictive of actions; SIGReg on the whole z. L = L_fwd + α L_inv + β L_adv + γ L_SIGReg, β = 1, γ = 0.09, α per task (3–10, Table I). Control uses **only z^c** in the CEM cost (eq. 7, 300 candidates × 30 iterations, 5-step plans, receding horizon 5). Baselines: LeWM, SMWM (inverse dynamics on the full latent instead of SIGReg).

### Protocol and results

- Nominal (Table II): CF-JEPA second or third everywhere; trails LeWM on Reacher (0.61 vs 0.86) and Push-T (0.78 vs 0.96) — the authors attribute it to the shared inverse-dynamics term with SMWM and, on Push-T, to SIGReg forcing information into z^u when nothing uncontrollable exists. **3 training seeds, ± std; baselines quoted from their papers.**
- Distractors: N coloured rectangles (49 px × 18–36 px) sliding along image edges at 3–18 px/step, never occluding the task. Models *trained* with N = 2 distractors. Table III (N = 2): Reacher LeWM 0.07 / SMWM 0.1 / **CF-JEPA 0.48**; Push-T 0.01 / 0.86 / 0.87; Cube 0.42 / 0.84 / 0.85; TwoRoom 0.33 / 0.98 / 0.95. Table IV as N grows: Push-T SMWM 0.86 → 0.27 at N = 64, CF-JEPA 0.87 → 0.57; TwoRoom SMWM 0.98 → 0.46 at N = 32, CF-JEPA 0.95 → 0.82. OGBench Cube is the exception: SMWM holds 0.73 at N = 128 because its action space is end-effector IK — perception of the arm is not required for control, so collapse does not matter.
- The perception-dependent test (Table V, ManiSkill reach with joint position control, N = 16, 3 seeds): at the loose 0.06 m threshold all similar; at 0.02 m CF-JEPA 0.35 vs SMWM 0.22 vs LeWM 0.21.
- Mechanism (Table VI): participation ratio of the latent on Reacher, nominal → distracted: LeWM 48.2 → 2.24 (21.5× collapse), SMWM 3.71 → 1.0, CF-JEPA 42.3 → 12.9 (3.3×). Decoded z^c shows the arm; decoded z^u shows blur (Fig. 5).

### What is not there

- Real robot: none; "applying it to a real robot would test the factorization under richer visual conditions."
- The split dimensions (128/64) are not ablated.
- Distractors that interact with the task (a person near the robot) are explicitly out of scope.

### For the SO-101

The §8 lighting angle in NEXT_ANGLES is this paper's distractor test with the distractor you actually have (lighting, background objects, your hand resetting the scene). The paper gives you the measurement to pre-register before any success rate: **participation ratio of W1 latents, nominal vs perturbed frames**. If W1 collapses under your perturbations the way LeWM does under theirs (48 → 2), that alone explains why surprise thresholds did not transfer to fresh rollouts (W1-A/C), and it is a one-evening measurement on data you already have. Note also the exception they found: factorisation only pays where control needs perception of the agent; your position-controlled arm with commanded joints available is closer to the OGBench-Cube case than the ManiSkill one, so the honest prior is "it will not change success, it may change monitor stability".

---

## 12b.7 Imagine-RL (Hu, Zhai, Zhao, Liang; 2609.24033)

### What it is

Latent-noise RL (DSRL-style) on a frozen flow-matching VLA, with a LeWM-style **visual–torque latent world model (VTLWM)** supplying the critic with imagined consequences of each candidate chunk. Nothing is fine-tuned in the VLA; the actor acts in the VLA's noise space; the world model is frozen during RL.

### Method

VTLWM: shared multi-view encoder → z_t; torque τ̂_t normalised and concatenated [z_t; τ̂_t]; action encoder; causal one-step predictor with history H = 15 predicting the next visual latent *and* torque. L_WM = MSE(visual) + λ_τ SmoothL1(torque) + λ_sig L_SIGReg (eq. 7). At critic time, rolled out H_f = 10 steps under the candidate chunk.

Critic: residual-confidence cross-attention (eqs. 8–12). The key device: the **previous-window prediction residual** R^m_{t,j} = X^{hist} − P^{prev} (what the model predicted last window vs what then happened), turned into a per-token confidence α = max(α_min, e^{−β ρ}) with ρ the relative error, and added as log α to the attention logits over *future* tokens — so a world model that has been wrong recently is trusted less on this candidate. The action critic Q_a(o, a; h^a) trains on executed chunks with TD targets (10-Q ensemble, mean); the noise critic Q_w distils it (eq. 4); the noise actor maximises Q_w.

### Protocol (§IV)

- Real PiPER, RealSense D435i, dual-arm collaborative; four contact-rich tasks: Socket Plug-in, Press-button, USB Plug-in, Peg-in-hole. Object pose randomised.
- π0 pretrained with 100 trajectories per task; RL methods get a 50-trajectory warm-up buffer, then 50 online episodes, updating every 10. Rewards: −1 per step, +5 success, −1 ordinary truncation, −50 severe failure. H_a = 30 chunk, first H_q = 10 executed.
- **50 trials per task** for evaluation, same object-pose ranges across methods. Baselines: π0, TA-VLA (torque-aware VLA), DSRL (same RL, no world-model critic features).
- Fig. 6: Socket 24 / 32 / 36 / **50**; Press 46 / 26 / 34 / **60**; USB 38 / 50 / 82 / **86**; Peg 62 / 62 / 68 / **76** (π0 / TA-VLA / DSRL / Imagine-RL). "+60 % relative over π0, +23.6 % over DSRL."
- World-model quality: cosine similarity of 10-step visual predictions 0.913 average (Table I); torque MAE 0.044 ≈ 1 % of range (Table II).
- Critic discrimination (Fig. 9): on 10 held-out successes + 10 failures per task, standardised ΔQ = E[Q | success] − E[Q | failure]: DSRL 0.975 → Imagine-RL 1.228 average (+26 %).

### What is not there

- No intervals on the 50-trial rates; no seeds.
- The residual-confidence gate is not ablated against plain attention over future tokens, so its contribution is unmeasured.
- Only 10 + 10 held-out trajectories for the discrimination analysis.

### For the SO-101

Two things are usable; the RL is not (no torque sensing, no fleet of 50 online episodes per task). First, the **critic-discrimination test** is a judge report card in disguise and the cheapest way to compare any two scorers on your 69 rollouts: standardise, ΔQ between labelled successes and failures, paired over episodes. Second, the residual-confidence idea is the principled version of your W1 surprise: not "abort when surprise is high" but "down-weight the imagined future when the last window's prediction was wrong". That is a 20-line change to the REAP head's input and the next monitor variant worth a bar.

---

## 12b.8 FOREWARN (Wu, Tian, Swamy, Bajcsy; CMU/Berkeley; 2502.01828)

### What it is

Policy steering: a Diffusion Policy proposes K plans; a world model imagines each plan's latent future; a VLM fine-tuned to **narrate the latent future in text** picks the plan that matches the task description. The split is the point: the world model does *foresight* (prediction), the VLM does *forethought* (evaluation in language), and neither is asked to do the other's job.

### Method

World model: DreamerV3 RSSM (not JEPA; included here because the pattern is model-agnostic), trained on 230 real trajectories per task (successes, failures, 100 demos; 300/30 split), predicting T = 64 future latents. VLM: Llama-3.2-11B-Vision with its image tokeniser **replaced** by the world model's encoder + dynamics + a linear projection into the text-embedding space; LoRA fine-tune on a VQA dataset of (latent sequence downsampled T/4, task description) → hand-written behaviour narration ("grasps the cup by the handle"). Steering (eq. 3): the same VLM is queried twice — narrate each of K = 6 NMS-aggregated plans, then choose the best plan given the task description, as multiple choice.

### Protocol

- Franka, 15 Hz, wrist + third-person cameras, 3D-printed gripper. Cup (handle / rim / interior modes) and Bag (edge / middle; deformable).
- Narration quality (Table I, 30 held-out rollouts per task, 3 fine-tuning seeds): GT accuracy (human-judged binary) FOREWARN 0.89 / 0.76 vs an **oracle with ground-truth future latents 0.92 / 0.78** vs VLM-Act (fine-tuned end-to-end from observation + action text) 0.37 / 0.34 vs GPT-4o on decoded images 0.34 / 0.27. The world-model future is nearly as good as the real future for narration; VLMs cannot read low-level actions or decoded frames.
- Steering (Table II, **10 trials per cell**): base policy 0.30 / 0.20; FOREWARN 0.80 / 0.70; a binary classifier on the same latents (Classifier-Dyn-Latent) 0.80 / 0.70 in-distribution but **0.00 / 0.10 under a novel task description** ("the handle is covered with oil"), where FOREWARN stays 0.80 / 0.70. VLM-DynLat-Category (VLM picks indices directly, no narration) 0.80 / 0.40 → 0.30 / 0.40 novel.
- Latency (Table III): WM prediction 0.1 s, narration 1.3 s, evaluation 2.3 s, total 3.7 s vs 22 s for VLM-Act.

### What is not there

- n = 10 per cell; the Wilson interval on 0.80 is roughly 0.49–0.94. Differences of 0.1 in Table II are not resolvable.
- The classifier baseline matching FOREWARN in-distribution is the result to remember: **narration buys generalisation to new instructions, not accuracy on the trained task.**
- Failures "stem from the world model's imprecise imagination" (App. B2) — the proxy is the bottleneck, as everywhere.

### For the SO-101

This is the v1.5 verifier-tier idea with a world model in front of the VLM, and the latency fits your harness (3.7 s at rest). The piece worth testing is narrow: can a VLM given W1's **imagined** latents (via a learned projection) say which of two chunks will nest the bowl, as well as the same VLM given the *real* final frames (your Flora/Qwen 8/8 result)? The oracle gap in Table I (0.89 vs 0.92) is the number you would be reproducing. The honest prior from 12a.8 and 12b.2: the imagined future carries the information only within the plannable range, so the test is on chunk-length futures, not whole episodes.

---

## 12b.8a LaLa-World (Song, Gu, Park, Nam, Maes, Heo, Han, Yun; NAVER AI Lab / Mila; 2610.27851, read 2026-10-24)

### What it is

A JEPA latent world model whose action input is **text**: numerical actions from seven control datasets are verbalised with fixed templates ("move the pusher 0.60 left and 1.95 down", "move gripper 0.5 cm forward … rotate roll +2 deg"), and image/video transitions without actions (OpenVid-1M captions by Qwen3-VL, OmniEdit and ShareGPT-4o edit instructions) enter the same objective with their caption as the condition. One predictor is trained across all of it. Lucas Maes (LeWM) is a co-author; this is the LeWM line scaled up, and the one any team building on LeWM is most likely to have read.

### Method

Frozen DINOv2-S/14 or DINOv3-L/16 encoder (no proprio, input or target). Predictor = **Qwen3.5-0.8B**, pretrained: visual patches through a 2-layer MLP aligner, language tokens through the LLM's own tokenizer, one learned query token per target patch with additive 3-D (t, h, w) sinusoidal embeddings, a 2-layer MLP head back to encoder dimension. Loss: plain MSE on the next latent, one step only, no SIGReg (the encoder is frozen, so collapse is not possible). Context 1–3 frames. Substeps are **summed** into one action per world-model step (10 → 2 dims on Push-T) and replayed as sum/5 at execution (App. D.4.2). Training: 8 × H100, 20 epochs of ~1 M samples, bf16, AdamW, lr 2e-4 cosine. Planning: CEM on squared latent distance to the encoded goal, verbalising each sampled action with the training template.

LaLa-Policy: the pretrained predictor plus action queries feeding a GR00T-style diffusion action head; predicts an action chunk and the future latent jointly, Cosmos-Policy style.

### Protocol

- Planning (Table 1, Table H): Wall, PointMaze, Push-T, MetaWorld-R/RW, Rope, Granular, DROID offline (Action Score from endpoint error), RoboCasa-R/P **zero-shot**. Evaluation trajectories 12–96 per benchmark; standard errors reported. **Planning horizon ≤ 6 steps everywhere.** Baselines (DreamerV3, TD-MPC2, IRIS, LeWM, DINO-WM, JEPA-WMs, V-JEPA 2-AC) retrained by the authors on the same data and planning settings.
- Latent vs generative (§4.2, Fig. 5): RAEv2 (DiT, flow matching) with the **same DINOv3 encoder, same language conditions, same data, same parameter count (0.8 B)**, 12 h budget; four depths for the time curve. This is the matched design of the paper.
- Data ablation (Fig. 4, C.7): control only vs control + image/video, with original and paraphrased templates.
- Initialisation (Table C): pretrained-LLM single-domain vs random-init joint vs pretrained joint.
- Language sensitivity (C.3, eq. 4): S(z) = 1 − ‖ê(z) − e_{t+1}‖ / ‖e_t − e_{t+1}‖ with the target fixed and z replaced by numerical / qualitative / **direction-reversed** / magnitude-swapped / empty; a predictor that reads its condition scores high on the first, negative on the reversed one.
- Policy (Tables 2–4): LIBERO (4 suites) and RoboCasa (50 human or 300 MimicGen demos), against Cosmos Policy fine-tuned by the authors without its value/proprio heads, plus random-init and Qwen-init LaLa-Policy.
- LLM actor in place of CEM (C.2, Table B): Claude Opus 5 and GPT-6 Astra propose N plans, receive only the latent cost, revise for K rounds.

### Numbers

- Table 1 sim average **72.1** vs JEPA-WMs 64.6, DINO-WM 62.3, **LeWM 28.1**; second table average 50.5 vs JEPA-WMs 44.8. PointMaze is a loss (56.2 vs 60.4).
- Fig. 5: **59.4 vs RAEv2 48.0** (5-step), reached in ~2 h vs RAEv2's 12 h (6×); full-depth planning 13.9× faster than 5-step RAEv2.
- Fig. 4: control-only 59.3 → all data 61.6 on original templates (**+2.3, about one SE**); on paraphrased templates 52.5 → 61.0 (+8.5). The robustness claim is supported; the planning-gain claim is not proven.
- Table C: pretrained single-domain 54.5, random-init joint 55.6, pretrained joint 60.2. Per domain, single-domain wins on Wall (97.9 vs 95.8) and PointMaze (66.7 vs 56.2); joint wins on Push-T, DROID and RoboCasa.
- Fig. F(a): random-init and pretrained Qwen reach **similar validation loss** but plan 55.6 vs 60.2; Nemotron-1B matches Qwen's loss yet barely responds to reversed directions (Fig. E(d)). Loss does not identify the useful predictor.
- Table A (ImageNet transforms): MRR 0.856 with language vs 0.026 with empty language vs 0.024 for the source latent — the predictor uses the condition.
- Table B: Claude Opus 5 as actor reaches 100 % on Wall with 90 rollouts (CEM 95.8 with 9 000) and 53.1 on RoboCasa-R with 80 (CEM 40.6 with 54 000); worse on PointMaze (31.2 vs 56.2).
- Policy: LIBERO average 97.7 vs Cosmos Policy 97.3, Qwen-init 95.7; RoboCasa 50 demos 57.3 vs 56.0, Qwen-init 56.8 (no intervals); inference 0.86 s vs 4.45 s per LIBERO trajectory.

### What is not there

No real-robot closed loop (DROID is offline endpoint error; Future Directions says so). No horizon beyond 6, so nothing here contradicts Planning Limits. No seeds on any number; standard errors are over evaluation trajectories only. The joint-training gain is not separated from the pretrained-LLM initialisation (single-domain *pretrained* vs joint *pretrained* is the comparison; it is mixed per domain). No commanded-action or proprio baseline in planning (proprio is excluded by design). Policy gains over Qwen-init are 0.5–2 points without intervals. LeWM's 28.1 is under this paper's protocol (no proprio, summed substeps, frozen-encoder baselines retrained by the authors); it is not LeWM's own number and should not be quoted as such.

### For the SO-101

Three things transfer, in order of cost:

1. **The sensitivity score is a free test of whether W1 reads its actions.** Same held-out transition, condition replaced by the true action, shuffled action from another episode, sign-reversed action, zero action. If S barely moves, W1's predictor ignores the action and G1b's failure is explained by that alone, before position control or horizon. It is TurboVLA's task-swap test for the world model. NEXT_ANGLES §28.
2. **Language as a calibration-agnostic action interface for pooling SO-101 Hub data.** Community SO-101 datasets share the joint space but not calibration offsets. Forward kinematics → end-effector deltas → template text would give one action format across them. Untested here on any arm with differing calibration; a proposal, not a result.
3. **LLM-as-planner over a latent critic (C.2)** is the shape of the v1.4 harness with W1 as critic, and the cheap version of §19 selection: Claude ranks N SmolVLA chunks by W1 cost. Simulation only, and worse than CEM on PointMaze; the closed-loop version needs the data engine's rollouts.

What does not transfer: the training scale (8 × H100, ~20 M samples), and the claim that video data improves planning (+2.3 within noise).

## 12b.8b HWM: Hierarchical Planning with Latent World Models (Zhang et al., CoRL 2026; 2604.03208; tested 2026-10-28)

### What it is

A two-level latent world model. The low level is an ordinary action-conditioned predictor. The high level (§2.1) encodes a chunk of primitive actions into a **macro-action** l (a small transformer, CLS token → R^32) and a predictor F2 jumps over the chunk in one step: (l, z_t) → z_{t+h}. The claim tested here is their Fig. 6: beyond roughly one second, one high-level step predicts the future latent better than rolling the low level out step by step — a **crossover** in prediction horizon. Their data: about 130 h of DROID and RoboSet.

### Protocol here (frozen `4643775`, before training)

W1's frozen encoder latents (192-d) on `so101_bowls_all_trimmed`, the G1b split (157 train / 25 val episodes); the high level trained on top, 20,000 steps, last step kept (no validation peeking). Three high levels identical except the action input: the learned macro-action, the concatenated primitive actions, and the net joint displacement. Bars: **H1** a crossover in [1.0, 2.0] s; **H2** high ≤ 0.95 × low at 2 s; **H3** learned macro-action ≥ net displacement on the cosine between predicted and true latent displacement. Total cost ≈ $0.4 on vast (cap $5).

One deviation, recorded before scoring: the first run read red-only frames (the bug in chapter 6.4a) and was declared invalid, not scored; the re-run's latents reproduce G0 (copy-last MSE at 30 frames 0.915 vs G0's 0.899).

### Numbers

Mean |ẑ − z| over the 192 dims, 700 validation anchors (± SE; anchors within an episode are correlated, so the SEs understate episode-level uncertainty):

| horizon | 0.73 s | 1.0 s | 1.27 s | 1.5 s | 2.0 s |
|---|---|---|---|---|---|
| low level (W1 rollout) | 0.382 | 0.438 | 0.481 | 0.511 | 0.563 |
| high level, learned macro-action | 0.769 | 0.766 | 0.761 | 0.764 | 0.759 |
| high level, net displacement | 0.790 | 0.793 | 0.789 | 0.793 | 0.799 |

H1 **fail** (no crossover: the high level is worse at every horizon); H2 **fail** (ratio 1.35 at 2 s); H3 **met** (cosine learned 0.608 vs displacement 0.551, about 6 SE).

Post-hoc, labelled as such in the README: predicting the train-mean latent scores 0.81 at every horizon, so the learned high level beats it by only 0.04–0.05 and is flat in the horizon; it beats copy-last only from 1.27 s on; its last-batch training loss is 0.15 against 0.76 on validation, which suggests overfitting on 157 episodes but does not measure it.

### Verdict: not proven, by the owner, against two failed bars

The bars say H1 and H2 fail. The verdict written is **not proven**, and the reason is chapter 2's scale rule: about 0.6 h of one task against the paper's ~130 h. The result fits "the high level needs far more data than the low level" and cannot separate it from "the crossover does not exist on this arm". This is the case to know cold, because it is the one an interviewer will push on: a pre-registered fail overruled to not proven. The defence is that the confound was nameable before the run (scale), that the bars were not moved, that the verdict word changed and the numbers did not, and that the experiment that would separate the two readings is named (more episodes, or the high level on a pretrained multi-task latent). The weakness to concede: the confound should have been written into the protocol as a conditional verdict, not applied by the owner afterwards. Chapter 5.4 carries it as a worked case.

### For the SO-101

Hierarchy is not a cheap fix for W1's long-horizon drift at this data scale. The H3 result is the one finding that transfers: a learned encoding of the action chunk carries more about the latent displacement than the net joint displacement does, which is a small piece of evidence that the action sequence, not only its endpoint, matters to the latent.

## 12b.8c SURVEYOR: certified speculative plan consumption (NeurIPS 2026 workshop, World Models in Physical AI; tested 2026-10-27)

### What it is

A planner proposes a sequence of latent subgoals (in the paper, FF-JEPA's ~53M diffusion drafter); the executor consumes them speculatively and accepts each step while the plan–execution discrepancy δ = ‖z_executed − ẑ_planned‖ / ‖z_executed‖ stays below a threshold τ, calibrated as the median relative distance between task-equivalent states (the paper's τ = 0.20). Above τ, it replans. Evidence: simulation only (PushT, Reacher, OGBench-Cube, Two-Room).

### Scope, stated up front

The drafter is not available, so this is not a reproduction. Two parts are reproducible on real data and were tested, with W1's own imagination as the drafter stand-in: (a) does τ have an operating point on real-robot LeWM latents; (b) does the accept rule work as a failure monitor. Protocol `02928b7`, frozen before any number.

### Numbers

Gate 0, τ on the 69 published rollouts (judge frames from G9c-3's frozen rule): τ = median relative distance between at-rest success frames of different episodes = **1.249** (the paper's 0.20). Non-equivalent pairs (success vs mid-episode, success vs failure end) have p10 **1.106**, below τ; equivalent-vs-non-equivalent AUROC 0.71. S1 (τ < p10 non-equivalent) **fail**, S2 (τ ≤ 0.40) **fail**. Two successful end states in different episodes are as far apart in W1's latent as a success and a mid-episode frame: the scene — bowl positions, lighting, arm pose at rest — dominates the latent, not the task state.

Gate 1, the accept rule as a monitor on the 50 primary episodes (42 ok / 8 fail), with a shuffled-drafter control: at S = 10 frames, 1/8 caught at 1/42 false interventions, AUROC 0.619 (shuffled 0.628); at S = 30, 3/8 at 6/42, AUROC 0.756 (shuffled 0.560), warnings 3.4 s **after** the labelled onset. S3 (≥ 5/8 at ≤ 2/42) **fail** at both; S4 (true ≥ shuffled + 0.05) met at S = 30 only.

### Reading

The accept rule is the **seventh latent-space monitor to fail on this arm at the standing bar**, after W1-A/B, W1-C1, W1-C2, G15, G16-B and the REAP head. Over 1 s the discrepancy carries signal beyond chance, but it fires after the failure has happened. Gate 0 is the more useful result: it is a direct measurement of *why* latent distance is not task distance on this rig — the same reason G6 (latent distance is not progress) and the PRIME graph check (13a.10a) failed.

### For the SO-101

No τ, no speculative consumption. If a subgoal-based planner is ever tried here, the acceptance test must be on a task-relevant subspace (object state, not the whole latent), and gate 0's measurement — equivalent vs non-equivalent end states — is the five-minute check to run first.

## 12b.9 How this family evaluates, side by side

| Paper | Platform | n per cell | Seeds / intervals | Matched control | Trivial / oracle baseline |
|---|---|---|---|---|---|
| JEPA-WM study | sim + DROID offline | 96 eps × last 10 epochs | 3 train seeds, std over epochs | one factor at a time from a base config | none |
| Planning Limits | Meta-World + BridgeData offline | 100 random alts; 20 paired eps/task | paired gain/loss counts | same predictor, 5 encoders; random-init predictor | **real simulator as perfect predictor** |
| D-JEPA | sim ×3, RoboTwin, **PiPER real** | 256 / 128 / 64 starts; real 50/task | Wilson CIs; paired bootstrap; disjoint fit/cal/eval | 5 re-implemented published objectives on same candidates | oracle (driving only) |
| PSG-JEPA | OGBench, LIBERO, **ALOHA real** | 200 goals; 1 500 rollouts; real 50/task | 3 planner seeds ± std; none on real | LeWM_ActionIDM built as control | none |
| SA+IDM | sim ×4 | 50 problems | 3 seeds, baselines quoted | IDM-only ablation | none |
| CF-JEPA | sim ×4 + ManiSkill | — | 3 seeds ± std, baselines quoted | SMWM | none |
| Imagine-RL | **PiPER real** | 50 trials/task; 10+10 for ΔQ | none | DSRL (same RL, no WM) | none |
| FOREWARN | **Franka real** | 30 narrations; **10 trials/cell** | 3 seeds on narration only | classifier on same latents | **oracle with true future latents** |
| LaLa-World | sim ×9 + DROID offline; RoboCasa zero-shot | 12–96 trajectories | SE over trajectories; no seeds | **RAEv2 with same encoder, data, conditions, size**; single- vs joint-domain | sensitivity score with reversed/empty condition |

Reading it with chapters 3–7:

- The two most convincing designs are the two with an **oracle**: Planning Limits (real simulator) and FOREWARN (true future latents). An oracle bounds what the learned component could ever add, and it is why those papers can say "the limit is the score, not the predictor" and "narration is near the ceiling". Ask for the oracle whenever one is constructible; on your arm the oracle for any imagined-future test is the *real* future frame, which you always have offline.
- Real-robot n is 50 per task in the three best cases and 10 in FOREWARN; none report seeds or intervals on hardware. That is the field's norm, and your register already exceeds it by bootstrapping episodes. Say so plainly, once.
- D-JEPA is the only paper in both chapters with disjoint fitting / calibration / evaluation identities and Wilson intervals in the main text. That is the standard to imitate for anything with a learned re-ranker or a tuned threshold.
- Matched controls are getting better: LeWM_ActionIDM (PSG), SMWM (CF-JEPA), DSRL (Imagine-RL), random-init predictor (Planning Limits) are all "same everything, remove the one idea". Still absent everywhere: the *non-learned* controller baseline — commanded joints, proprio distance, episode length — which is the one you keep finding wins.

## 12b.10 Corrections and additions to chapters 12 and NEXT_ANGLES

- 12.5 D-JEPA line: expand. It is a bounded, permutation-equivariant set operator trained on **executed outcomes of the whole candidate set**; your proxy-label test is a different experiment and must be renamed in NEXT_ANGLES §1 (e.g. "learned re-ranker on heuristic labels") or redesigned with ~8 executed candidates per start from the G7 cube rollouts.
- 12.4.1 planner: add the Planning Limits numbers (real simulator 92 → 41 %) as the mechanism behind G1b, alongside the position-control argument; the fix the paper proves is a near subgoal within the rollout, not a better model.
- 12.5 PSG line (§7): the pre-registered design is three-armed (LeWM / +ActionIDM / PSG) at matched epochs; expected gain is in transition grounding (Δq), and the judge is the G7 object-state probe, not open-loop MSE.
- 12.5 CF-JEPA line (§8): register the participation-ratio measurement (nominal vs perturbed frames) as the first, pre-success-rate gate; prior is "monitor stability, not success".
- 12.2 (collapse): add the fourth failure mode, slow-feature collapse under scene variation (H-JEPA App. L; CF-JEPA Table VI), and that inverse dynamics is the standard fix but is itself ill-posed as a transition target (PSG §3.3).
- 12.4.3 monitors: add residual-confidence weighting (Imagine-RL eq. 10) as the next REAP variant.
- New candidate protocol, cheapest in both chapters: **VLA+WM selection** (Planning Limits §7) — SmolVLA proposes N = 8 chunks, W1 scores against a demo subgoal 0.25 s ahead, paired episodes, pooled bar; plus the random-init-W1 control that the paper ran (0.8 %).

- 12.4.1 planner: add LaLa-World's sensitivity score as the first check before any further planning gate — a predictor that ignores its action cannot rank chunks (§28). And quote LeWM's 28.1 in Table 1 only with its protocol caveat.
- 12.3 / 12c: Table C and Fig. F(a) are a fourth independent case of equal loss, different planning (random-init vs pretrained Qwen; Nemotron vs Qwen).

- 12.4.1 planner / hierarchy (12b.8b): HWM's crossover does not appear at ~0.6 h of one task; verdict not proven on the scale confound. Any hierarchical model on this data must report the train-mean and copy-last baselines beside the low level.
- 12.4.3 monitors (12b.8c): SURVEYOR's accept rule is the seventh latent monitor to fail the standing bar; gate 0's equivalent-vs-non-equivalent end-state distances (τ 1.249 vs non-equivalent p10 1.106) are the measurement behind "the scene dominates the latent".

## 12b.11 Exercises

1. Write the Planning Limits ranking test for your 69 rollouts: K = 7 (one chunk), L ∈ {7, 14, 28, 56} frames, 100 perturbed-demo negatives per point, expert percentile per L. Report the plannable range of W1 and of the commanded-joint baseline on the same negatives. Decide beforehand what P\* would have to be for any W1 ranking gate to be worth running.
2. State D-JEPA's bounded-correction property (Prop. 2) in one sentence and derive the condition under which the operator cannot change the argmin. Then count, from the G7 runbook, how many executed candidates per start you could afford in one robot day.
3. Implement PSG's dynamic grounding head on W1 (Δq over k = 1…3) as a training-only addition; before training, write the three-arm protocol with the matched-epoch control and the G7 probe as judge; commit it.
4. Compute the participation ratio and r_95 for W1 latents on 256 nominal frames and 256 frames from the perturbed sessions. One table, one paragraph on what it predicts for W1-A/C.
5. From the Imagine-RL discrimination analysis, write the ΔQ report-card format for your judges and apply it to G9c-3, Flora and the REAP head on the same 8 + 42 episodes.
6. Compute LaLa-World's S(z) for W1 on 200 held-out transitions from the 69 rollouts with true / shuffled / reversed / zero actions; before running, write what value of S under the reversed action would make you stop every W1 ranking gate.
