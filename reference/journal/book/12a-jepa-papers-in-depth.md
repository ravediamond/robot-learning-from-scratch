# Chapter 12a. The JEPA world-model papers, in depth

Chapter 12 says what a JEPA world model is and what your gates found. This chapter is the reading behind it: the eight papers in the knowledge base that define the family, each read from the method and experiment sections rather than the abstract, with the objective written out, the training setup, every evaluation protocol with its metric and trial count, the headline numbers with table references, what the authors did not run, and what each result means for a position-controlled arm with two webcams.

The reason to know these at this depth is not trivia. An interviewer at a world-models company will ask "how did LeWM evaluate planning?" or "what is RoboJEPA's scaling law actually of?" and the difference between a research engineer and a reader is whether the answer contains the protocol: how many trials, against which baseline, with what held fixed.

The papers, in the order they build on each other:

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | DINO-WM | 2411.04983 | Frozen DINOv2 patches + ViT predictor; CEM planning to an image goal; the baseline everyone else beats |
| 2 | LeWorldModel (LeWM) | 2603.19312 | End-to-end from pixels, SIGReg against collapse, 15M params, Push-T 90 % |
| 3 | V-JEPA 2.1 | 2603.14482 | The big encoder; dense loss; 10-trial Franka planning |
| 4 | RoboJEPA | 2610.10515 | Scaling law for a predictor on frozen V-JEPA 2.1 features, 22M–8B |
| 5 | H-JEPA | 2610.06805 | Hierarchy of LeWMs across timescales; AntMaze 18 → 73 % |
| 6 | Temporal-Distance-JEPA | 2607.25337 | Mine a directed cost from step order; when to plan with it vs latent ℓ2 |
| 7 | DeepJEPA | 2610.00368 | Depth inside a transition as a test-time axis; the CEM elite-boundary argument |
| 8 | ContactGuard | 2608.13438 | LeWM-style model as a pre-contact failure monitor; the paper closest to your G10/REAP |

Companion reading in the next family (chapter 12b, planning and grounding): Planning Limits of JEPA, D-JEPA, physical grounding, CF-JEPA, one-step fidelity (2610.19375).

---

## 12a.1 DINO-WM (Zhou, Pan, LeCun, Pinto; 2411.04983)

### Method

The model is three parts, only one of which is trained for control:

- **Observation model**: frozen DINOv2, image 224×224 → N patch embeddings z_t ∈ ℝ^{N×E}. Not fine-tuned, ever. The paper's thesis is that a general pretrained patch representation removes the need for task-specific visual coverage.
- **Transition model**: a ViT with the tokenisation layer removed (a decoder-only transformer over patches). Input: history of H latent states and actions; action a_t goes through an MLP and is **concatenated to every patch vector**; proprioception, when present, the same way. Causal attention is at the **frame** level: every patch of z_t attends to all patches of z_{t−H:t−1}, and the whole next frame is predicted at once (contrast IRIS, which predicts patch by patch autoregressively within a frame; ablated in their A.4.2).
- **Decoder**: transposed convolutions trained separately with ‖q(z_t) − o_t‖², for visualisation only. Backpropagating its loss into the predictor was tried and hurt (A.4.3).

Objective (their eq. 1), teacher-forced over segments of length H+1:

  L_pred = ‖ p_θ(enc(o_{t−H:t}), φ(a_{t−H:t})) − enc(o_{t+1}) ‖²

No reconstruction, no reward, no termination. Because the encoder is frozen, collapse is impossible by construction, which is why DINO-WM needs no SIGReg/EMA (chapter 12.2): the price is that the representation cannot adapt to the task.

### Planning

Goal is an RGB image o_g. Cost C = ‖ẑ_T − z_g‖² after rolling the predictor from enc(o_0) under candidate actions. Optimiser: CEM inside MPC. They also tried gradient descent through the differentiable model and found CEM better (A.5.3) — a result repeated in every later paper in this family (H-JEPA is the exception: it plans by gradient descent, with planner compute reported in TFLOPs).

### Evaluation protocol

Six environments: PointMaze, Wall, Reach (robot arm), Push-T, Rope, Granular (XArm, deformables). Metric: **success rate on 50 sampled (initial, goal) pairs** for Maze/Wall/Reach/Push-T; **Chamfer distance on 10 instances** for Rope/Granular (environment stepping is slow). Push-T goals sampled to be feasible within 25 steps. Baselines all trained on the same offline data without reward: IRIS, DreamerV3, TD-MPC2, and AVDC (diffusion video model) qualitatively.

Headline numbers (Table 1): Push-T SR 0.90 (DreamerV3 0.30, IRIS 0.32, TD-MPC2 0.00); Reach 0.92 (DreamerV3 0.64); Granular CD 0.26 (DreamerV3 1.05). Encoder ablation (Table 2): patch features matter — Push-T with DINO CLS 0.44, ResNet 0.20, R3M 0.42, DINO patches 0.90. The CLS-vs-patch gap is the single most important number in the paper for you: a global vector loses the spatial detail that manipulation needs. Generalisation to unseen layouts (Table 3): WallRandom 0.82 vs DreamerV3 0.76; PushObj (unseen shapes) 0.34 — low for everyone. LPIPS of decoded rollouts (Table 4) 0.007 on Push-T vs 0.045 R3M.

### What is not there

- No standard errors anywhere; 50 instances once, no seeds reported in the main text.
- No real robot. Every environment is simulated.
- No "commanded-action" or "no-world-model" baseline: the question "does planning through the latent beat a trivial controller" is not asked. Your G1b asked it, and on your hardware the trivial controller won.
- TD-MPC2 at 0.00 everywhere is reported honestly as "the lack of reward signal makes it hard to learn good latents", i.e. a baseline run outside its design regime.

### For the SO-101

Your W2 was this recipe (DINOv2 patches, lost; W2′ is the retrain). Two lessons transfer directly. First, the CLS/patch result says that if W2′ is used for anything object-level (G7), it must use patch tokens, not a pooled vector. Second, DINO-WM's 48-step CEM on Push-T took 47 s per plan in LeWM's measurement; on an Orin the frozen DINOv2-B encoder alone is hundreds of ms per frame, so DINO-WM-style online planning is off the table and the role is offline ranking or verification. That is the hardware fact behind chapter 12's "verifier, not planner".

---

## 12a.2 LeWorldModel (Maes, Le Lidec, Scieur, LeCun, Balestriero; 2603.19312)

### Method

End-to-end from pixels, no frozen encoder, no EMA target. Encoder: ViT-tiny, patch 14, 12 layers, 3 heads, width 192, ≈5M params; CLS token → a one-layer MLP projector with BatchNorm gives a single 192-d latent per frame. Predictor: 6 layers, 16 heads, 10 % dropout, ≈10M params; action conditioning by **AdaLN with zero init** (so at initialisation the predictor ignores actions and learns to use them); causal masking over an N-frame history. Total ≈15M.

Objective:

  L = ‖ ẑ_{t+1} − z_{t+1} ‖² + λ · SIGReg(Z),  λ = 0.1

where SIGReg (from LeJEPA) is the anti-collapse term: project the batch of latents on M = 1024 random directions and apply a univariate normality test (Epps–Pulley) to each projection; by the Cramér–Wold theorem, if every 1-d projection is Gaussian the joint is Gaussian. Pushing the latent toward an isotropic Gaussian prevents both collapse to a point and collapse to a low-rank subspace, which is what chapter 12.2 means by "collapse, and three fixes" (EMA target, VICReg-style variance/covariance, SIGReg).

Note that there is **no stop-gradient on the target**: the same encoder encodes z_{t+1}, and SIGReg alone keeps it from the trivial solution. That is the paper's claim ("stable end-to-end").

### Planning

CEM with terminal cost ‖ẑ_H − z_g‖², MPC replanning. Horizons short.

### Evaluation protocol

Push-T, Two-Room, Reacher, OGBench Cube (plus Wall and others in the appendix). The main planning figure (Fig. 6) is **success at matched training FLOPs**: Push-T 90 % for LeWM vs 13 % for DINO-WM at the same compute, with LeWM's full model 48× faster to plan (0.98 s vs 47 s per episode). Linear-probe table (Table 1): correlation r between a linear readout of the latent and ground-truth state on Push-T: agent position 0.974, block location 0.986, block angle 0.902 — the latent does contain object state in simulation. A violation-of-expectation test: physically impossible transitions raise prediction error, used as a qualitative "surprise" demonstration.

Baseline count: DINO-WM and pixel-space models, at matched FLOPs. Seeds: planning success is reported over episodes; standard deviations appear in some tables, not all.

### What is not there

- Real robot: none.
- A non-learned baseline for the planner.
- Long horizons: the authors list "short planning horizons" as the limitation; H-JEPA is the follow-up.
- Any failure-detection use: ContactGuard adds that.

### For the SO-101

W1 is this model (192-d, SIGReg, your two cameras). Everything in chapter 12.4 is LeWM's recipe applied to real data: G0 showed the dynamics are learned (0.31 vs copy-last 0.89); G10 showed the probe result transfers (outcome readable at rest, AUROC 1.0); G1b showed the planning result does **not** transfer on a position-controlled arm where the commanded joints already determine the outcome. The Table 1 probe is the exercise to repeat on your own frames with object position as the target (chapter 12.7 exercise 2 and G7): LeWM's r = 0.986 for block location is the number you are checking against.

---

## 12a.3 V-JEPA 2.1 (Meta FAIR; 2603.14482)

### Method

The encoder family, not a small model. Two changes over V-JEPA 2:

- **Dense loss** L = L_predict + L_context. L_predict is the usual masked prediction (predict EMA-target features of masked tubes from visible context, L1). L_context additionally predicts the **visible** tokens' target features, down-weighted by λ_i = λ/√d_min(i, M) where d_min is the distance to the nearest masked token — so near-mask context tokens get almost full weight and far ones almost none. The effect is dense per-token supervision instead of mask-only.
- **Deep self-supervision**: the same loss applied at four intermediate encoder depths, so lower layers are also trained to be predictive.
- A multi-modal tokeniser so the same encoder takes image, video and depth.

Training: 135k iterations + 12k cool-down; ViT-L to ViT-G.

### Evaluation protocol

Almost entirely representation benchmarks (ADE20K segmentation, depth, SSv2, EK100 action anticipation, Ego4D STA), with the ablation ladder in Table 1: ADE20K mIoU 22.2 (V-JEPA 2) → 33.8 (+context loss) → 38.6 (+multi-level) → 40.8 → 41.4 → 47.1 (ViT-G) → 47.9. That ladder is the paper's evidence: each change isolated, one metric.

The robot part (Table 6): Franka, action-conditioned predictor (300M) post-trained on DROID, CEM 800 samples × 10 iterations, H = 1, image goals: Reach 100 %, Grasp 70 %, Pick-and-place 80 %; with 15 iterations and H = 8, Grasp 80 %. **n = 10 trials per skill**, one robot, one lab. Navigation (Table 7): 10× faster planning than NWM at equal success.

### What is not there

- n = 10 means a Wilson 95 % interval on 70 % is roughly 40–89 %. The robot table is a demonstration, not a comparison; the paper does not claim otherwise.
- No baseline policy on the same robot tasks.
- The predictor is post-trained on DROID (Franka). Nothing about transfer to a different embodiment.

### For the SO-101

The ViT-L encoder at 256 px is 526 ms per frame on the Orin (your RoboJEPA measurement), which rules it out of any loop under 1 s. Its place in your stack is offline: feature extraction for a judge or a dataset-level probe. The dense-loss idea (supervise visible tokens near the mask) is a training trick that would apply to W1 only if you masked, which you do not. The thing to take from this paper is the ablation ladder in Table 1 as a model of how to report: one change per row, one metric, cumulative.

---

## 12a.4 RoboJEPA (Zholus, Assran et al., Meta FAIR; 2610.10515)

### Method

A deterministic next-feature predictor on top of a **frozen V-JEPA 2.1-G** encoder (D_e = 1664, patch grid H/16 × W/16). Tokenisation: every patch projected to width D; action a_t and proprio s_t each projected by a robot-id-specific linear layer; per timestep the sequence is (visual tokens of all views, a_t, s_t). Attention: joint space-time-view, **timestep-causal**, and **local to 8 timesteps** so it never attends to a longer history than it trained on (that is what lets it roll out autoregressively past the training length). 4-axis RoPE (t, i, j, v) plus a learnable per-layer view bias so view order can be randomised. QK-norm "of particular importance for stability when scaling data and model".

Objective (eq. 5): two L1 losses averaged —

  L_TF = Σ ‖ẑ_{t+1} − z_{t+1}‖₁ (teacher-forced, one step)
  L_AR = Σ ‖z̃_{t+1} − z_{t+1}‖₁ (autoregressive from a prefix, with a KV-cache, over the remaining frames)
  L = ½ L_TF + ½ L_AR

The AR term is the exposure-bias fix: the model trains on its own predictions. Proprio is not predicted; it is integrated (s_{t+1} = s_t + a_t) or zero for absolute commands.

Data: 23 datasets, 12 embodiments, 26 action spaces, 15,022 video hours, 6,692 action hours (DROID, RoboSet, RoboMind, **LeRobot community data 21.5k episodes with SO-101**, RoboCasa365, 1X, AgiBot World, Open-X). GPUs partitioned into four groups by (views, resolution) with per-group batch sizes.

Training: AdamW, wd 0.1, clip 1.0, warm-up → flat 5e-5 → linear decay to zero; the flat stage is shared across compute budgets and only the cool-down is re-run per budget (the WSD trick). Cool-down recipes: short-horizon (K = 2 rollout) and long-horizon (K = 10, mixed 240×320 and 480×640).

### The scaling law

Metric: per-token L1 between predicted and encoder features averaged over a **9-step latent rollout on held-out scenes**, two views (one side + wrist), initial observation + recorded actions. Two eval sets: DROID holdout (real Franka, unseen scenes) and RoboCasa (sim). Compute C counted exactly in FLOPs. Compute-optimal frontier = lower envelope across model sizes. Four candidate forms fit on 22M–2B, judged by **held-out extrapolation error at 4B and 8B** (Table 2): plain power law 2.0 / 3.7 ×10⁻³; second-order L(C) = E + A·C^{α−γ ln C} 0.6 / 1.4 ×10⁻³ (adopted); BNSL 1.0 / 1.8; UNSL 1.4 / 5.7. Fitted irreducible error E ≈ 0.22 (DROID) / 0.17 (RoboCasa) and the frontier is close to it — "close to saturation under the current data budget".

That is the real content of "scaling law" here: an **extrapolation test**, with the law chosen by how well it predicts the two sizes held out of the fit. That is the standard to hold any scaling claim to.

### Downstream protocol

- RoboCasa greedy tasks (pose reach; reach while holding): closed-loop planning, normalised simulator return, Fig. 6. Capability emerges ≈10²⁰ FLOPs (50–100M models). Saturates early.
- Non-greedy tasks (obstacle-reach, object-push): with the short-horizon recipe, **no clean frontier** (300M beats 1B, Fig. 7); with the K = 10 cool-down, a clean frontier appears, obstacle avoidance takes off at 10²¹ and saturates 10²²; object-push success is 0 for every model until ≈10²² FLOPs (Fig. 9). Order of emergence: end-effector control → holding an object → static scene geometry → object dynamics.
- Real Franka (DROID rig, side + wrist): Grasp, Object Lift, Pick-and-Place; single final image goal; same planning hyper-parameters for all sizes; "over 50,000 evaluation episodes" across both platforms in total. Table 3 (progress / success %): RoboJEPA-8B Grasp 67/67, Lift 65/50, P&P 42/27; 4B 60/60, 54/30, 49/21. π0.5 as a *reference, not a baseline* (text goal, DROID-finetuned): Grasp 25/5, Lift 12/0, P&P 69/53. Qualitative: only ≥ 2B keep the gripper closed after grasping.
- Planning: receding-horizon CEM, uniform action proposals, rolling KV-cache, precompiled inference, multi-GPU — "several seconds per step for 8B, not real-time".

### What is not there

- Per-task n on the real robot is in Appendix G, not the main text; no intervals in Table 3.
- The law is for a dynamics model on a fixed representation; encoder+predictor jointly is explicitly out of scope.
- No learned proposal distribution; the planner samples uniformly.
- No language.

### For the SO-101

Your arm's data is in the training mix (LeRobot group, 21.5k episodes). Your measurement that action conditioning works on your rollouts (shuffled-action error 9–10 % above aligned) is the G0-style check at the far-left end of their curve. The encoder cost (526 ms) is the wall. The idea to take is the two-term loss: W1 trains teacher-forced only; adding an AR term over a short prefix is a one-evening change and the measurement is the 25-step imagination error that REAP already logs. And the method lesson: their non-greedy result (no frontier until training horizon matched planning horizon) is the same train/plan mismatch Temporal-Distance-JEPA names, seen from the compute side.

---

## 12a.5 H-JEPA (Zhang, Terver, Rabbat, LeCun, Balestriero; 2610.06805)

### Method

Level 1 is LeWM. Level ℓ > 1 has its own state encoder E^(ℓ) (pointwise 2-layer MLP over the level-(ℓ−1) latent, window w = 1), its own action encoder A^(ℓ) that pools s_ℓ lower-level action embeddings, and its own causal-transformer predictor F^(ℓ); stride s_ℓ = 2 at every level, so level 3 predicts 4 env steps per step. Each level has a LeWM loss ‖F^(ℓ)(z_{<τ}, a_{<τ}) − z_τ^(ℓ)‖² + λ SIGReg(Z^(ℓ)), and **all levels train end-to-end**, gradients from upper losses flowing into lower encoders.

On DROID an **inverse-dynamics term** is required (eq. 8): I^(ℓ)(z_t, z_{t+1}) → sg[a_t], because with scene variation across episodes the purely predictive objective collapses to "encode the background, which is predictable, and drop the arm, which is not" (their appendix L shows this slow-feature solution is always feasible when the regulariser only sees the marginal). Without IDM on DROID, planning fidelity is zero.

### Planning

Top-down: level L optimises macro-actions by **gradient descent** to minimise ‖ẑ_{H_L} − g^(L)‖²; its predicted states become subgoals for level L−1 (eq. 6: terminal term + β × intermediate subgoal terms, matched in the upper level's latent space via E^(ℓ+1)); level 1 emits primitive actions. Closed loop: execute a prefix, re-encode, replan.

### Evaluation protocol

Simulated: FourRoomDistractors, Visual AntMaze, OGBench Cube, Push-T. Metric: success rate; **three paired model+planner seeds, error bars SE**; planner compute swept and reported as **TFLOPs per episode** (Fig. 6 top is success vs planner FLOPs, a Pareto front); depth sweep (1–4 levels) at a fixed 100-TFLOP budget on a separate set of 50 tasks per environment (Fig. 6 bottom). Baselines: flat LeWM; HWM (same hierarchy but a shared latent space: upper encoders are identity) to separate "temporal decomposition" from "distinct abstract spaces".

Numbers: AntMaze 18 % → 73 % at three levels with less planner compute; two-level matches or beats LeWM on Push-T, three-level fails on Push-T because episodes are too short (three-level sees 58 % of LeWM's transitions per epoch, four-level 14 %, Table 3). HWM beats LeWM everywhere (temporal decomposition alone helps); H-JEPA beats HWM most on AntMaze (distinct spaces help where there is a frequency gap).

Mechanism experiments, which are the model of what "read the failure" means at paper scale:

- Probe NMSE per entity vs level (Fig. 4): AntMaze body pose becomes unrecoverable at higher levels while position stays recoverable; the distractor in FourRoom fades. The **entity-frequency gap** (ratio of spectral centroids of fastest and slowest state components) predicts when this selective abstraction appears (Fig. 5): > 2× gap → abstraction; Push-T, Cube, DROID have gaps < 2× and show none.
- "Change only the cost" (Table 1): flat level-1 planning but with the goal distance measured after projecting through upper encoders raises AntMaze success 23 → 31 (L2 projection) with the level-1 dynamics unchanged; so part of the gain is the cost geometry, not the hierarchy of planners.
- Monotonicity of cost along expert trajectories (Fig. 8): Spearman 0.89–1.00 for subgoal tracking vs 0.48–0.75 for full-horizon goal cost.

DROID: no simulator, so **Fréchet fidelity** (eq. 7): (d_F(0, p̄) − d_F(p, p̄)) / d_F(0, p̄), the planned end-effector path vs the expert's, normalised so that doing nothing = 0 and the expert = 100 %; 15 clips, three train seeds. LeWM+IDM ≈ +34 pts over collapse; H-JEPA two-level +40 at lower planner budget; frozen-encoder baselines (JEPA-WM, V-JEPA 2-AC) need orders of magnitude more compute for lower fidelity (Fig. 9c).

### What is not there

- Closed-loop real robot: none ("a next step").
- The Push-T three-level failure is attributed to data, not tested by training longer.
- Fréchet fidelity is an open-loop proxy; its calibration against success is in an appendix, in simulation.

### For the SO-101

G6b was your hierarchy test and failed on loops. H-JEPA explains why that was the wrong place to test: your task has no frequency gap (one arm, one object, ~1 Hz everything), and the paper's own Fig. 5 predicts no selective abstraction on exactly that kind of data. The transferable pieces are the IDM term — your W1 trains on single-scene data so it does not need it, but W2′ on mixed lighting (§8 angle) might — and the Fréchet-fidelity idea: an offline, open-loop plan-quality metric against expert paths on your 69 rollouts, which costs nothing and would have told you before G1b that the plans do not follow the demos.

---

## 12a.6 Temporal-Distance-JEPA (Bai, Xiong; 2607.25337)

### The diagnosis

Fig. 1: on held-out Push-T demonstration pairs, LeWM latent Euclidean distance vs true step gap has Spearman ρ = 0.65; a cost mined from the same logs ρ = 0.95. JEPA trains one-step prediction; MPC needs a multi-step ranking; the ranking is inherited from embedding geometry "as a by-product". That sentence is the train–plan gap, and it is the same gap your G6 found (latent distance is not progress).

### Method

Keep LeWM entirely (encoder, predictor, SIGReg). Add a **directed cost head** d_ψ(z_s, z_g) with the metric-residual (MRN) form (eq. 5): ‖φ_sym(z_s) − φ_sym(z_g)‖₂ + max_k ReLU(φ_asym,k(z_s) − φ_asym,k(z_g)) — non-negative, asymmetric by construction (going A→B need not cost B→A). Supervise it (eq. 6) with **same-trajectory step counts** τ(i,j) = j − i as regression targets (Smooth-L1) plus a **cross-trajectory hinge** pushing d_ψ(z_i, z_{g'}) above a margin for goals from other trajectories at the same time index. Add a **horizon-matched rollout-consistency loss** (eq. 8): H = 5 steps of teacher-forced open-loop prediction, the same operator the planner uses. Full objective eq. 9: λ_roll 1.0, λ_td 1.0, λ_sig 0.09, prediction window 5.

Training details (App. C): ViT-tiny ≈15M, image 224, 192-d latent, history 3, AdamW 5e-5, wd 1e-3, bf16, 10 epochs; Push-T effective batch 128.

### Evaluation protocol

This paper's protocol section is the best-written in the family and the one to copy:

- **Locked evaluation**: checkpoint, episode manifest (50 episodes), solver (CEM-30 or iCEM-30, 300 candidates, H = 5, goal offset 25), budget all held fixed; **ten independent seed runs, mean ± std** for the primary comparisons (Table 2); three seeds for ablations (Table 4), stated whenever it changes.
- Table 2 (locked, ten seeds): Two-Room LeWM 97.4±1.3 → TD-JEPA 100.0±0.0; Reacher 96.0±1.9 → 97.0±2.4; Push-T 83.6±2.3 → 86.0±4.2; OGB-Cube 68.0±2.8 → 82.2±2.9 (+14.2, the headline). RC-aux as a second baseline.
- Table 3 places the same locked numbers beside published point estimates and says in the caption "rows are not strictly comparable" — the correct way to show literature context without claiming a head-to-head.
- Ablations (Table 4, Push-T, three seeds): remove the MRN form → 56.7 (from 82.7 blend); remove cross-trajectory hinge → 72.0; remove rollout consistency → 62.0.
- **Plan-cost comparison with everything fixed** (Fig. 3, Table 11): on navigation pure d_ψ ≥ ℓ2; on Push-T pure d_ψ 69.0 vs ℓ2 86.0 — the mined cost *loses* on contact tasks even though it ranks time better (Table 6: ρ(d_ψ, τ) 0.914 vs ρ(ℓ2, τ) 0.785). Table 7 locates the failure: d_ψ fails post-contact (14.3 of 15.6 failures) where fine pose matters. Table 9: a contact gate (d_ψ far, ℓ2 near) does not recover ℓ2's score; the planner needs geometric scoring throughout.
- Table 10: cost aggregation (terminal-only w = 1 vs mixed w = 0.3) changes Reacher from 74 to 96 for the same checkpoint — a sensitivity check that would have been a confound if unreported.

### What is not there

- Real robot: none.
- The two-role rule (d_ψ for topology, ℓ2 for contact) is chosen per environment by the authors, not learned.

### For the SO-101

Three things. First, "ranks time better but plans worse on contact" is the exact shape of your G6 (progress-by-distance dropped) and your G1b: a cost that is monotone along demonstrations need not be the cost that separates good from bad chunks near the object. Second, the locked-protocol vocabulary — manifest, seeds, solver fixed, "not strictly comparable" — is what your register does with gate IDs; say so in the interview. Third, the cross-trajectory hinge is a cheap negative-mining trick for any progress head you train on your 69 rollouts.

---

## 12a.7 DeepJEPA (Jin, Zhang et al.; 2610.00368)

### Method

Same LeWM planner, but the transition is a **weight-tied recurrent cell** F_θ applied k = 1…K_max times (K_max = 4) within one imagined step (eqs. 1–3), with context from the last L = 3 latents and actions. A linear **continue head** p_k = σ(H_φ(h^{(k)})) is trained with BCE on the label y_k = 𝟙[r_k > τ_rel], where r_k = (e_k − e_{k+1}) / max(e_k, ε) is the relative error reduction from one more update against the *training-time* target latent (τ_rel = 5e-4). At test time the target is unavailable; the head halts at the first depth with p ≤ η. Loss: L_final + λ_inter L_inter + β L_cont. CEM is unchanged: same candidates, elites, iterations, horizon.

### The argument

Proposition 3.1: for a CEM iteration with elite margin γ_k (cost gap between worst elite and best non-elite), if the largest per-candidate cost correction from one more update satisfies 2‖δ‖_∞ < γ_k, the elite set, and hence the next proposal mean and covariance, are unchanged. So extra depth can only matter when a candidate sits near the elite boundary. That predicts: outcome flips are sparse, depend on the sampled search population, and need not coincide with better state decodability. The experiments test those three predictions.

### Evaluation protocol

Reacher, Cube Single/Double/Triple, Push-T. **Three training/evaluation seed pairs × 50 episodes = 150 per cell**; fixed K ∈ {1,2,3,4} and learned halting share the same recurrent checkpoint, episode roots and CEM seeds; LeWM is a separate non-recurrent family. They report the best threshold from a full η sweep and say so ("characterises the achievable frontier rather than held-out threshold selection").

Table 1: uniform depth is not monotone — Reacher K1 84.0, K2 82.0, K3 83.3, K4 82.7; Push-T K4 is worst. DeepJEPA: Reacher 85.3, Double 74.0, Triple 77.3, Push-T 11.6, Single 79.3 (= K1), with mean depth 1.00–1.26. Table 2: the η sweep shows the operating point is task-dependent and that *lower* mean depth can give higher success (Cube Double 1.78 → 1.26 improves it).

Where depth goes (Fig. 4): on Push-T, transitions refined beyond K = 1 go from 1.24 % before contact to 9.02 % at contact onset (95 % cluster-bootstrap CI [5.01, 13.88]); on Cube Triple sustained contact +12.5 pts ([2.8, 22.6]), release −9.9 ([−14.6, −5.1]), adjusted for rollout step, distance and action norm. **The head saw no contact labels.**

How it changes decisions (Fig. 5): 93–99 % of 150 roots have the same outcome at K1 and K4; Cube Double 106 succeed at both, 34 fail at both, 3 helped, 3 harmed. Exact-batch replay reproduces 297/300 labels. Across 450 root–seed pairs on Cube Triple, K4 − K1 = +1.1 pp, CI [−1.3, +3.6], paired exact p = 0.424 — i.e. **the net effect of uniform depth is zero and they say so**. Re-sampling the search on the ten discordant roots flips direction in 5 of 30 trials. Probes (Q5): adaptive refinement changes pusher/object position error by 0.0105 and 0.0036 px on 10,000 held-out Push-T transitions — no decodability gain.

### What is not there

- Real robot: none. The headline gains are 1–3 pp in simulation with three seeds.
- The threshold is chosen on the evaluation sweep, by their own statement.

### For the SO-101

This is a method paper more than a result paper, and its method is the chapter-8 method: a mechanism (elite boundary), three testable predictions, each tested with a paired design, cluster-bootstrap CIs, exact replay to rule out logging mismatch, and a probe to rule out the tempting alternative explanation. Note how a +1.1 pp result with p = 0.424 is presented as evidence *for* the mechanism (sparsity), not hidden. The thing not to do on the Orin is implement DeepJEPA: your planner is not the bottleneck, your cost is.

---

## 12a.8 ContactGuard (Zheng, Johnson-Roberson, Zhi; 2608.13438)

### Method

The closest paper in the family to what you actually run. A LeWM-style action-conditioned latent world model as a **pre-contact execution monitor** for a chunked policy (ACT), policy-decoupled: it only sees the observation and the chunk the policy already emitted.

- World model: shared ViT-tiny per camera, **mean-pool the V = 3 view latents** then a learned linear projection → one latent z_t ∈ ℝ^D (so the dimension is independent of the number of cameras); shared action embedder; predictor = 4 AdaLN-zero transformer blocks + 2-layer MLP; trained teacher-forced with the LeWM loss (eq. 4): (1/L) Σ ‖ẑ_{i+C} − z_{i+C}‖² + λ L_reg over a context of C frames and L targets per sample; "each training target matches one inference-time rollout step". Trained on **unlabelled** rollouts and teleop; then frozen.
- Probe: anchor at t = T_g − k_pre (T_g = first gripper-command frame crossing the closure threshold, k_pre = 15 frames = 0.5 s at 30 Hz), roll out **K = 30 steps** under the recorded/planned chunk (0.5 s past closure), standardise ẑ_{t+K} with training-split statistics, and fit a **class-balanced ℓ2-regularised logistic regression** (eq. 6) on ≈250 labelled grasp attempts per task. Threshold τ selected once per task on validation and frozen.
- Online: trigger (first open→close transition in the upcoming chunk) → anchor → rollout → P(fail) > τ → abort before contact.

### Evaluation protocol

Robot: 14-DoF AgileX Piper dual-arm, three synchronised RGB cameras, ACT policy. Four grasp settings: cup, box (pick-and-place), pencil (pencil-and-notebook), towel (towel-fold). Test episodes collected separately and excluded from world-model training, probe fitting, threshold selection and ablation tuning.

- **Closed-loop (Table 1)**: N = 50 monitored live rollouts per setting; when the monitor would abort, the robot pauses 5 s with the gripper open and then resumes to record the would-be outcome ("a controlled counterfactual proxy"). Metrics: precision, recall, **false-abort rate FP/(FP+TN)**, ROC AUC, balanced accuracy. Ours vs single-view LeWM vs Direct-linear (probe on anchor latent + chunk, no rollout): Cup AUC 0.992 vs 0.928 vs 0.661; Box 0.946/0.933/0.619; Pencil 0.898/0.872/0.732; Towel 0.917/0.794/0.841. FAR for Ours 0.12/0.107/0.071/0.24.
- **Offline pools (Table 2)**: n = 101/126/103/144, five seeded stratified 5-fold CV, mean ± std; FAIL-Detect, RND, SAFE, **Current latent** (same encoder, labels, probe, pools, CV — only the imagined future replaced by the current latent), ContactGuard. Cup .982±.003 vs current .840±.008 vs SAFE .840; Box .984 vs .893; Pencil .992 vs .923; Towel .978 vs .871. Paired bootstrap 95 % CIs on the ContactGuard − Current-latent difference exclude zero on every task.
- **Action-corruption (Table 3)**: replace the planned chunk with shuffled (other episode), zero or mean actions at K = 30: Cup full 0.965 → shuffled 0.535, zero 0.847, mean 0.842. Shuffled collapses to chance on all four — the signal is the *specific* action's imagined consequence, not "this state looks risky".
- **Counterfactual action-swap**: hold observation fixed, swap in a chunk from a failed attempt at the same trigger: P(fail) rises +0.25/+0.33/+0.56/+0.65; the action-free current-latent probe is invariant by construction.
- **Proprio ablation (Table 5)**: adding the 28-d state *hurts* on all four (Cup 0.660 vs 0.920 image-only); read as a small-data shortcut, "we do not claim proprioception is harmful under richer fusion".
- **Runtime (Table 4)**, RTX 5090 FP32: K = 30 rollout 15.4 ms, full encode+rollout+probe 19.2 ms; K = 5: 6.5 ms.

### What is not there

- Recovery: it aborts and stops; no replanning.
- Only grasp-closure as the trigger; longer-horizon skills "would require hierarchical or repeated event-level monitoring".
- The world model's own prediction quality is never reported (no G0-style number); the probe AUC is the only evidence it learned anything.
- A policy-identity or episode-length baseline for the failure labels is not run.

### For the SO-101

This is the paper to measure your monitor work against, and you already have: **W1-C1** (2026-10-19) is this protocol on W1 — anchor 15 frames before first close, 30-step imagination under the executed chunk, logistic probe, current latent as the matched baseline, 5-fold × 5 seeds. Result: imagined 0.70, current 0.82. Imagination is 0.12 *worse*, the opposite sign of ContactGuard's Table 2 (0.98 vs 0.84). Candidate reasons, none proven: their world model trains on unlabelled rollouts *and* teleop from the deployed policy's distribution (yours on teleop only); three views mean-pooled vs two; ~250 labelled grasps per task vs 65 scored episodes; a torque-controlled arm whose pre-contact approach carries information a position-controlled one does not. The two controls that would say which, both offline and free: ContactGuard's Table 3 action-corruption test on W1-C1 (shuffled / zero / mean chunks — if the imagined-latent AUROC does not fall below 0.70, the rollout carries no action information and the encoder, not the probe, is the gap), and the G0 prediction-error split by pre- vs post-contact frames. Until they run, the honest sentence is "ContactGuard's design did not transfer to W1; the mechanism is unknown." Latency scales: their 19 ms on a 5090 is roughly your 153 ms per 25-step imagination on the Orin.

---

## 12a.9 How this family evaluates, side by side

| Paper | Platform | Unit | n | Seeds / intervals | Matched baseline | Trivial baseline |
|---|---|---|---|---|---|---|
| DINO-WM | sim ×6 | success / Chamfer | 50 / 10 instances | none in main text | IRIS, DreamerV3, TD-MPC2 (same data) | none |
| LeWM | sim | success at matched FLOPs; probe r | episodes | partial | DINO-WM at equal compute | none |
| V-JEPA 2.1 | benchmarks + Franka | mIoU…; success | **10 trials/skill** | none for robot | ablation ladder (Table 1) | none |
| RoboJEPA | DROID holdout, RoboCasa, Franka | L1 rollout error; success/progress | >50k episodes total; per-task n in appendix | extrapolation test on held-out sizes | π0.5 as reference only | none |
| H-JEPA | sim ×4, DROID offline | success vs planner TFLOPs; Fréchet fidelity | 50 tasks / 15 clips | **3 paired seeds, SE** | HWM (shared latent) | zero-action path (fidelity = 0) |
| TD-JEPA | sim ×4 | success, locked manifest | 50 episodes | **10 seeds, ±std**; 3 for ablations | LeWM, RC-aux, same checkpoint different cost | none |
| DeepJEPA | sim ×5 | success | 150/cell | **3 seed pairs; cluster bootstrap; paired exact p** | fixed K on same checkpoint | — |
| ContactGuard | real Piper | AUC, FAR, precision/recall | 50 live/task; 101–144 offline | 5×5-fold CV ± std; paired bootstrap | **current latent, same everything** | shuffled / zero / mean actions |

What the table says, in the words of chapters 3–7:

- Only two of eight run a real robot closed-loop; one of those at n = 10. "Real-robot result" in this family mostly means a demonstration.
- The cheap-baseline habit (chapter 3) is rare: no paper computes "what does a controller without the world model achieve" except as a reference VLA; ContactGuard's action-corruption and current-latent columns are the closest thing, and they are why its claim is believable.
- The strongest protocols (TD-JEPA, DeepJEPA, H-JEPA) are all about **holding things fixed**: checkpoint, manifest, solver, seeds, with the one varied factor named in the table caption. That is chapter 5.
- Nobody pre-registers; DeepJEPA says openly that its threshold is picked on the sweep. Your register is stricter than the field on that axis, and you can say so without arrogance by pointing at their sentence.
- Metrics drift toward proxies when the real thing is expensive: L1 rollout error (RoboJEPA), Fréchet fidelity (H-JEPA), linear-probe r (LeWM). Each paper that uses a proxy also shows, somewhere, that it tracks the real thing (RoboJEPA Fig. 6 vs Fig. 5; H-JEPA App. J.2). Ask for that figure whenever someone shows you a proxy. 2610.19375's r ≈ −0.01 between one-step error and planning success is the counterexample that makes the question necessary.

## 12a.10 Corrections to chapter 12 from this reading

- 12.5, H-JEPA line: "tested as G6b, failed on loops" stands, but add: H-JEPA's own Fig. 5 predicts no selective abstraction on data with an entity-frequency gap below 2×, which your loop data is; the test was outside the paper's claimed regime.
- 12.5, RoboJEPA line: the 9–10 % action-conditioning margin is a G0-style check, not a scaling result; the paper's scaling claim is about a predictor on frozen features, validated by extrapolation to 4B/8B, and it explicitly excludes joint encoder training.
- 12.4.3, monitors: add ContactGuard as the positive reference; the design differences (event anchoring, action-conditioned rollout, linear probe on imagined latent, current-latent control) are the next protocol, not a reason the role is closed.
- 12.2: LeWM has no EMA and no stop-gradient; SIGReg alone. V-JEPA 2.1 has EMA target + stop-gradient and no SIGReg. Keep the two recipes distinct when you explain collapse.

## 12a.11 Exercises

1. Write TD-JEPA's full objective (eq. 9) from memory, then LeWM's; mark the two terms that differ and say which planner operator each one supervises.
2. Re-derive DeepJEPA's Proposition 3.1 in four lines. Then state what it predicts about your G1b: with p^7 candidates and the 0.133 ceiling, how large an elite margin would a per-candidate cost correction have to cross?
3. Compute H-JEPA's Fréchet fidelity for the G1b plans against the 69 demonstrations' joint paths (use joint space; you have no end-effector). Report mean and three-seed-free bootstrap over episodes. Decide whether it would have predicted G1b.
4. Write the ContactGuard protocol for your arm in the NEXT_ANGLES format: trigger, anchor k_pre, K, probe, the current-latent control, the shuffled-chunk control, n per cell, bar. Do not run it until the 25 onsets are labelled.
5. For RoboJEPA, state in one sentence what the scaling law is a law *of*, and the test that chose its functional form.
