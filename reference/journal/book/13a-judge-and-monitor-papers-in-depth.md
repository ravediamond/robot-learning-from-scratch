# Chapter 13a. The judge and monitor papers, in depth

Chapter 13 says what judges and monitors are for and what your gates found (G9c-3 at 0 FP; Robometer beaten by episode length; Flora 8/8 with whole-episode input; the REAP-style head at 6/8 with two confounds). This chapter is the reading behind it: ten papers, each read from the method and experiment sections, with the objective, the training setup, every evaluation protocol with its metric and n, the headline numbers with table references, what was not run, and what each means for an arm whose only sensors are two webcams and six joint encoders.

One frame to hold while reading: the No Free Checker survey's Table 5. A verifier has four uses (curation, ranking, training reward, runtime gate), the error it must not make differs for each (a false positive trains on a failure; the wrong checkpoint ships; the policy satisfies the verifier without doing the task; hardware damage), and only the last two expose the score to a search for where it is wrong. Every paper below sits in one or two of those rows, and its evaluation is only as good as its match to that row.

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | FAIL-Detect | 2503.08558 | Learned score + time-varying conformal band; failure data not needed; logpZO wins |
| 2 | SAFE | 2506.09937 | Multitask: a probe on the VLA's own features generalises to unseen tasks |
| 3 | ReAP | 2610.16309 | Failure-supervised visual representation + action-conditioned predictor; the paper your head is named after |
| 4 | FARM | 2609.11445 | 34k-parameter readout on frozen VLA-JEPA states; SO-101 in the real-robot table |
| 5 | Robometer | 2603.02115 | 1M-trajectory reward model; progress + preference; the model length beat on your rollouts |
| 6 | STDR | 2606.31377 | VLM-segmented stages → stage + progress reward for RL from scratch; grasp gate against reward hacking |
| 7 | FailSafe | 2510.01642 | Inject failures in sim, verify recovery actions, fine-tune a VLM to emit ΔA |
| 8 | VLA-Corrector | 2609.06508 | GRU verifier on observable history + stage-routed recovery prompt to a frozen π0.5 |
| 9 | Spotter | 2609.36808 | Policy leads, VLM screens in parallel, intervenes on confirmed error; 70 % less time than VLM-led |
| 10 | No Free Checker | 2609.09250 | Survey of ~150 verifiers; availability vs credibility; nine metrics for a checkable claim |

---

## 13a.1 FAIL-Detect (Xu, Nguyen, Agia et al., TRI; 2503.08558)

### Method

Two stages, both calibrated only on **successful** rollouts: (1) a scalar score s_t from the policy's observation embedding; (2) a **time-varying conformal band** on s_t fitted to the empirical success distribution, so a flag is raised when s_t exceeds the band at t. The learned scores: **logpZO** (a flow-matching model from the embedding distribution to N(0, I); the score is the log-density of the embedding under that pushforward, i.e. an OOD score computed by one forward integration), **NatPN** (natural posterior network, an evidential-uncertainty density), **RND** (random-network distillation error). Post-hoc baselines: **STAC** (statistical distance between consecutive predicted action chunks on the overlap; needs 256 policy samples per step), PCA-kmeans distance, SPARC smoothness.

The design choice that matters: no failure data at training time, so the detector can be built the day the policy exists. The cost: every OOD-but-successful state is a false positive, and §VI shows exactly that on hardware.

### Protocol and numbers

- Simulation: four robomimic tasks (Square, Transport, Can, Toolhang) × two policies (Diffusion Policy, flow) × ID and OOD (camera bump) = 16 settings; calibrated on successful ID rollouts; **top-1 count across the 16** is the summary (Table II / Fig. 4): logpZO 10/16, RND 5/16, STAC 3/16 among post-hoc, PCA-kmeans never.
- Hardware: Franka bimanual, FoldRedTowel and CleanUpSpill, ID / ID+Disturb / OOD; **lower rollout counts than sim** (Table III); logpZO top-1 on 8/12 (balanced accuracy and weighted accuracy); PCA-kmeans 4/12; RND never top-1 but always top-3.
- Detection time: logpZO fastest in 3/8 sim settings; **STAC's detection time exceeds average success duration** (useless as an early warning).
- Compute: logpZO 0.04 s / 0.033 s per step on an A6000; STAC 1.45 s (36–44× slower).
- Fig. 7: score spikes coincide with the gripper losing the square (t = 72), the hammer slipping (t = 320), the second towel fold failing (t = 3648).
- Hardware caveat (§VI): with ID-only bands, OOD-but-successful rollouts are flagged as failures "at times yielding low or close to zero TNR"; the fix is setting-dependent calibration, which needs successes from the new setting.

### What is not there

- Intervals: none; the "top-1 in k/16" summary hides margins.
- A policy-identity or episode-length baseline.
- Failure-labelled training (that is SAFE's contribution).
- The paper's own limitation (§VII): the learned scores "focus on simpler robot state information (e.g. gripper closed or open) over the higher-dimensional visual features".

### For the SO-101

W1-surprise with a conformal band on successes is this paper's recipe with the world model as the score, and it failed the same way the paper's hardware section describes: ID-calibrated bands do not transfer to fresh rollouts (W1-A/C: thresholds fit on one session, 2/8 caught on the next). The paper names the cause (the success distribution shifts with the setting) and the cost of the fix (successes from every new setting). The transferable piece is the **time-varying band**: your thresholds were flat; a per-timestep quantile on the 42 successes is a one-evening change and the honest re-test of W1-A.

---

## 13a.2 SAFE (Gu, Ju, Sun, Gilitschenski, Nishimura, Itkina, Shkurti; UofT/TRI; 2506.09937)

### Method

The claim: a VLA's internal features already separate success from failure in a task-generic way (Fig. 1: t-SNE of π0-FAST features on LIBERO-10 shows one "failure zone" shared across tasks). So train a tiny probe on them. Input: the hidden state e_t from the VLA's last layer before action decoding, aggregated into one vector (ablated in appendix). Two heads: **SAFE-MLP** f(e_{0:t}) = Σ_τ σ(g(e_τ)) with L1 loss pushing failures up and successes down; **SAFE-LSTM** σ(LSTM(e_{0:t})) with BCE. Both trained on successful **and** failed rollouts from the seen tasks with trajectory-level labels only (no onset annotation). Threshold: **functional conformal prediction** on successful calibration rollouts, one-sided band upper_t = μ_t + h_t, FPR ≤ α on any time step of a new success.

### Protocol

- Simulation: LIBERO-10 (OpenVLA, π0, π0-FAST; 3 of 10 tasks random unseen), SimplerEnv (π0 reproduction; Google Robot and WidowX, 3 seen / 1 unseen); **3 random seeds over different seen/unseen splits**.
- Real: π0-FAST-DROID on a Franka, 13 tasks, **30 successes + 30 failures per task**, 3 unseen; OpenVLA on WidowX, 8 tasks, 532 rollouts (244 / 288), 2 unseen. No fine-tuning of the VLA.
- Baselines, all adapted to VLAs by the authors: token uncertainty (max/avg prob, entropy), embedding distance (Mahalanobis, k-NN Euclidean/cosine, PCA-kmeans), learned OOD (RND, logpZO trained on *both* classes, difference of scores), sample consistency (total variance of 10 sampled chunks, cluster entropy), STAC and STAC-Single.
- Metrics: ROC-AUC on the max-so-far score s̄_T (a flag anywhere on a success is a false positive, so the max is the right statistic); then at a CP threshold, TPR / FPR / balanced accuracy / normalised detection time T-det, traded off by α (Fig. 4).

### Numbers

- Table 1 (sim, 3 seeds): average unseen ROC-AUC SAFE-LSTM 77.04, SAFE-MLP 78.00 vs best baseline ~73.9 (Mahalanobis seen 80.28 / unseen 67.45); the sample-consistency methods are competitive but need 10 policy samples per step (π0-FAST 221 % slower).
- Real (Fig. 6): SAFE-MLP best on both robots; Mahalanobis / k-NN second.
- Fig. 4: on OpenVLA+LIBERO and π0+SimplerEnv the SAFE curves dominate the bal-acc vs T-det frontier; on π0-FAST+LIBERO, STAC is competitive at the cost of 10 samples per step.
- Overhead: < 1 ms, < 1 % of π0 inference.

### What is not there

- Intervals: seeds are averaged, no std in Table 1; the real-robot figure has none.
- Coverage shift: the detector is trained on one policy's failures; nothing about re-checking when the policy improves (your 2610.24801 note).
- A length or policy-identity baseline.

### For the SO-101

This is the P² / "probe the policy's own activations" idea from chapter 13.2 done properly, and it is the right first monitor for you because it costs nothing at runtime. SmolVLA in vla.cpp exposes the action-expert input features; a logistic probe on them, trained with the 8 + 42 labels and the functional CP band, is a day's work. The protocol to copy is the **seen/unseen split with three random splits** and ROC-AUC on the max-so-far score. The cheap baseline the paper did not run — episode length — is yours to add, and on your data it has beaten every learned judge so far.

---

## 13a.3 ReAP (Liu, Qing, Wei, Hou; ACE Robotics / Fudan; 2610.16309)

### Method

Two stages. **Stage 1 (detector)**: frozen DINOv2 patches from each camera → shared projection + camera and spatial embeddings → Perceiver-style cross-attention from M = 32 learnable latent queries → mean-pool → linear → compact state z_t → head → r^det_t ∈ [0, 1]. Class-weighted BCE (eq. 1) with the **ramp target** y_{i,t} = 0 on successes and t/(T_i − 1) on failures — no onset annotation; early failure frames get low targets. **Stage 2 (predictor)**: with the detector frozen, action-chunk tokens (per-action projection + step embedding) cross-attend to L_t; mean-pool; concatenate with z_t; a residual MLP predicts a latent update → ẑ_{t+H|t} → the frozen head → r^pred_{t+H|t}. Loss (eq. 2): (λ_feat/d) ‖ẑ − sg(z_{t+H})‖² + λ_risk BCE(r^pred), λ = 1 / 1. Alarm = detector OR predictor over their separately calibrated thresholds (eq. 3). Calibration (§4.3): on held-out successes; absolute / normalised / staged time parameterisations × pointwise quantile or functional conformal = six variants, appendix E.

### Protocol

- Offline: LIBERO-10 (OpenVLA 53.6 % SR, π0 80.4 %), RoboTwin-8 (π0.5 48.4 %, Fast-WAM 66.6 %); rollouts collected from the policy, held-out successes for calibration, evaluation split separate; **averaged over five α ∈ {0.05, …, 0.25}; mean ± std over three seeds** (Table 1).
- Primary metric **bTWA** (eq. 4) = ½ [ (1/N_fail) Σ_{detected} (1 − t_i/T_i) + N_TN / N_succ ]: earliness-weighted recall averaged with specificity; plus bACC and T-det.
- Baselines in three families: action-uncertainty (STAC, Diff-DAgger, ACCEL), OOD (RND, logpZO, FIPER), classifier (SAFE-MLP/LSTM).
- Real: three tasks (Fold Cloth 60 %, Stack Plates 75 %, Pick Cubes 50 % SR) with separately fine-tuned π0.5; monitor runs live, **alarms recorded without altering execution**, baselines scored post hoc on the same rollouts (Table 2a). Latency on an A800: detector 9.72 ms (no policy inference needed), predictor 81.16 ms including the 67.07 ms policy pass = +14.09 ms.

### Numbers

- Table 1: bTWA OpenVLA/LIBERO ReAP 0.701 ± 0.013 vs SAFE-MLP 0.622, RND 0.584; π0/LIBERO 0.780 vs 0.758 / 0.751; π0.5/RoboTwin 0.701 vs 0.667 (RND); Fast-WAM 0.651 vs 0.578 (SAFE-LSTM). T-det 0.326 / 0.395 / 0.306 / 0.592. "Up to 12.7 % relative over the strongest baseline"; the detector alone is already above every baseline, the OR adds 0.02–0.03 and lowers T-det.
- Table 2a real: ReAP bTWA 0.603 / 0.818 / 0.718, bACC 0.713 / 0.900 / 0.800; SAFE-MLP wins bTWA on Fold Cloth (0.624).
- Mechanism (§5.4, Table 3): generic PCA of DINOv2 as prediction space 0.644; **random (untrained) detector 0.650**; learned detector 0.683 — the failure-supervised space is worth 0.03–0.04 bTWA, and mean-pooling frozen DINOv2 already beats every baseline (0.635). Fig. 3b: latent-transition loss alone 0.493, risk loss alone 0.642, both 0.683. Action conditioning (Table 3b): zero actions 0.520 / 0.541, **shuffled chunk from another rollout of the same task 0.618**, full 0.683; Table 4: shuffling raises L2 10.35 vs 9.68.

### What is not there

- The real-robot table has no seeds or intervals; n per task is in appendix C.4.
- Onset labels: the ramp target avoids them, which is a strength, but it means T-det is measured against the episode length, not the true onset — a failure that begins at 90 % of the episode is scored as "detected late" for any monitor.
- An episode-length or policy-identity baseline.
- Coverage shift when the policy changes.

### For the SO-101

Your REAP-style head v1 (logistic on [W1 latent, commanded actions, step], 6/8 at 7/42) was a stripped version — and v2 (same day, ramp target, no step feature, common-horizon cut, shuffled-chunk control: 3/8 at 5/42, AUROC 0.789, +0.08 over shuffled) showed what the stripping had hidden: the time feature and episode length were the 0.93. The analysis below was written before v2 and is kept because each point was tested by it: (a) your latent is W1's, trained on prediction, not a failure-supervised space — Table 3a says that costs ~0.03–0.04 and that a *random* projection of DINOv2 features is almost as good as their learned one, so the encoder is not where your gap is; (b) you had Gemini onsets and a step feature; they use the ramp target and no onsets at all — adopt the ramp target, it removes the onset confound from your verdict entirely; (c) they report bTWA, which combines earliness and specificity in one number your bar (≥ 5/8 at ≤ 1/20) approximates crudely. The decisive borrowed control is **Table 3b**: train your head once, then evaluate with shuffled chunks from other episodes; if the number does not drop, the head is not reading the action, and "action-conditioned" comes out of the name. Also: their detector needs no policy inference (9.7 ms) — on the Orin that is the W1 encoder pass you already pay.

---

## 13a.4 FARM (Pei, Luo, Wang, Lv, Guo, Zhong, Ci; CASIA; 2609.11445)

### Method

The narrowest claim in the chapter and therefore the cleanest: do the **unaltered** internal predictive states of a frozen VLA-JEPA (the 768 × 1024 world-state tokens after the 12th predictor block, pre-norm, action tokens removed) already contain failure information that a **33,985-parameter** readout can decode? Readout: shared Linear 1024→32 + LN + GELU per token; attention pooling with a 32-d query; MLP 32→32; sigmoid → s_t. Task-balanced BCE with trajectory-level labels (hierarchical sampling: task, trajectory, step). Causal risk q_t = max_{τ≤t} s_τ. Two deployment modes: fixed-readout zero-shot (source θ* on target WM states) and readout-only adaptation (fine-tune the 34k parameters on labelled target trajectories).

### Protocol

- Simulation: 500 LIBERO trajectories (246 S / 254 F) over 10 tasks, 50 each; 7 seen (S1–S7) / 3 strict-unseen (U1–U3); **each task truncated to its common horizon T_k so length cannot be a shortcut** (Table I); fixed split 30 train / 20 eval per seen task; **five-fold out-of-fold** trajectory-grouped CV with seeds 0/1/2 for the representation and causal analyses.
- Baselines (15, matched split/horizons/metrics): SAFE-MLP/LSTM retrained; Mahalanobis, k-NN, PCA-kmeans; RND, logpZO; action total/translation/rotation/gripper variance; cluster entropy; STAC, STAC-Single. Representation controls: best single low-order statistic, 7-stat linear, predictor-input tokens, block-6 tokens.
- Real: PIPER X / π0.6* (497 external-policy trajectories), PIPER X / VLA-JEPA (100), **SO-101 / Eval-RL (70; blue-cube grasping; 10 Hz, 8-frame two-camera history, one score per checkpoint)**, Franka / π0-FAST-DROID (SAFE's 778-rollout public population). Three readout lineages: Core (350 source trajectories), Expanded (446, different source population), Adapt (readout-only on target, OOF).
- Metrics: AUROC / AUPRC, macro (per-task average) and pooled.

### Numbers

- Table II: five-fold OOF pooled 85.68 / 88.59; best single statistic 72.49 / 76.33; 7-stat linear 77.19 / 78.81; predictor-input tokens 83.10 / 85.89; block 6 82.41 / 85.58. So the failure information is in the WM state, is more than low-order statistics, and is concentrated near the output.
- Table III (10-task, seen): FARM 83.62 / 85.55 macro vs SAFE-MLP 78.20 / 80.67 (+5.42 / +4.88); strict-unseen: **STAC-Single wins zero-shot (67.12 / 73.35) over FARM (64.88 / 69.04)**; Adapt-35 (35 target trajectories per task): FARM 74.23 / 75.40 vs SAFE 69.15.
- Table IV real (pooled AUROC / AUPRC): PIPER X π0.6 Expanded ZS 84.70 / 79.13, Core 41.35 / 35.43, Adapt 98.48 / 98.41; PIPER X VLA-JEPA 90.91 / 93.20, 72.73 / 90.11, 95.73 / 97.80; **SO-101 Eval-RL 69.85 / 68.46, 57.80 / 42.76, 82.88 / 73.06**; Franka 55.89 / 55.82, 51.58 / 54.66, 75.78 / 74.31.
- Fig. 5 (causal partial history): at 25 % of the trajectory pooled AUROC 75.07 ± 0.73; at 75 % 81.63 ± 0.40, 1.78 below full. Fig. 2: 1 → 7 source tasks lifts pooled AUROC 61.7 → 89.7.
- Latency: 0.2256 ms mean CUDA on an RTX 5090 once the WM state exists.

### What is not there

- The SO-101 row is n = 70 from an RL policy, three-fold; no intervals printed; the zero-shot Core number (57.8) is near chance and the authors say so ("zero-shot transfer is not uniform").
- What the policy executing on the SO-101 was, beyond "in-house RL".
- The time cost of the VLA-JEPA forward that produces the WM state (526 ms-class encoder on your Orin).

### For the SO-101

A paper with your arm in its real-robot table is worth reading twice. The honest reading: a readout on frozen VLA-JEPA states reaches 0.83 AUROC on an SO-101 grasping task **after adaptation on ~47 labelled episodes**, and 0.58–0.70 zero-shot. Your G10 probe on W1's at-rest latent reached 1.0 on the bowl task with far fewer labels, which says the two tasks are not comparable (grasp success mid-episode vs nesting at rest), not that W1 is better. What transfers: (1) the **common-horizon truncation** (Table I) as the length-shortcut control you have been doing by hand — adopt it; (2) **macro and pooled** both reported; (3) the partial-history curve (Fig. 5) as the right way to report "how early" without onset labels; (4) the result that predictor-output tokens beat predictor-input tokens by ~3 points — the one-line check of whether W1's *predicted* latent carries more than its encoded one, which is the premise of every imagination-based monitor you have tried.

---

## 13a.5 Robometer (Liang, Korkmaz, Zhang, Hwang et al.; USC/UTD/MIT/UW/NVIDIA; 2603.02115)

### Method

Qwen3-VL-4B-Instruct with learned tokens inserted in the sequence: ⟨prog_token⟩ after each frame of video 1 (progress and success heads, causal so frame t sees only ≤ t) and one ⟨pref_token⟩ after both videos (preference head). Three losses: L_pref (binary which-is-better), L_prog (cross-entropy over N = 10 progress bins against p_t = t/T, continuous value as expectation), L_succ (balanced BCE, s_t = 1 only at t = T). T = 8 frames by fixed-length subsampling that always includes first and current frame. Preference pairs built three ways (§III-D): different expertise (expert vs unlabelled failure of the same instruction), **instruction negatives** (same frames, wrong instruction → progress 0), video rewind (chosen forward vs rejected rewound segment, with decreasing progress targets on the rewind).

Data: RBM-1M, 1M trajectories, 21 embodiments (Open-X, AgiBot World, Epic-Kitchens, RH20T, LIBERO, failure-detection datasets); **SO-101 is not in it** (the offline-RL experiment says so).

### Protocol and numbers

- Q1 reward evaluation on RBM-EVAL-OOD (976 trajectories, 3 institutions, 6 embodiments, 3 not in RBM-1M) and -ID: **VOC Pearson r** of predicted reward vs frame index on successes; **Kendall τ_a** of final reward vs the (failed < suboptimal < successful) ordering within a task. Table I: Robometer 0.92 / 0.94 VOC, τ 0.64; RoboReward-8B 0.82 / 0.88, 0.47; GVL 0.16 / 0.21. Fig. 3: instruction confusion matrix diagonal mean 0.54 vs RoboReward-8B 0.28.
- Table II: zero-shot on RoboFAC (11k failures/corrections) VOC 0.652, τ 0.436, success−fail diff 0.141; LoRA fine-tune 0.875 / 0.786 / 0.271; fine-tuning the base Qwen3-VL from scratch is *worse* than zero-shot Robometer on 2/3.
- Ablations (Table III, LIBERO-90, 8,262 held-out pairs): progress only τ 0.63; + preference 0.74; + failed data 0.92 and success−fail diff 0.11 → 0.46; ReWiND architecture instead of a pretrained VLM 0.48 / −0.14.
- Q3 downstream: online RL (DSRL on a DROID π0 base) 20 → 85 % single-stage, 20 → 70 % two-stage vs RoboReward 55 % / 20 % (**20 evaluation trials**); offline IQL on SO-101 noisy+expert: Robometer best, "2.4× over the best baseline" (Fig. 7: Oven 0.85 vs sparse 0.50; Plate 0.15 vs 0.05); retrieval 100 queries, 5 tasks (Fig. 8); zero-shot failure detection on 100 DROID trajectories (30 S / 70 F), 7 tasks, F1 Table IV: Robometer 0.81, RoboReward 0.74, GPT-5-mini 0.33, token uncertainty 0.48.

### What is not there

- Intervals anywhere; 20 trials for the RL claims; the SO-101 offline-RL plate task at 0.15 vs 0.05 is 3 vs 1 successes in 20.
- **Episode length** as a baseline for the failure-detection F1 or the τ ordering. Your G9b showed length at 0.938 vs Robometer 0.818 on ArmnetBench and length winning all four held-out tasks; the paper never computes it.
- The failure-detection rule ("temporal inconsistencies in predicted per-frame rewards") is not specified in the main text.
- The authors' own limitation (§V): no access to contact, grasp stability or compliance; "may fail to recognise or correctly reward certain failure cases".

### For the SO-101

Your G9b result and this paper agree once the question is phrased right. Robometer answers "how far along does this look and which of two looks further" and it is good at that (VOC 0.92). Your failure mode is identity (wrong bowl), and a trajectory-comparison objective with instruction negatives at the *dataset* level has no reason to catch it at the *frame* level on a task it never saw; Table IV's "pull tissue" and "fold towel" rows (0.58, 0.76) are the paper's own version of the same gap. The lesson for any reward-model gate: (1) length first; (2) an instruction-negative test *on your frames* (same episode, swap the instruction, does the score drop?) before trusting its language grounding; (3) report τ on the three-way ordering only when you have suboptimal-but-successful episodes, which you do not yet label.

---

## 13a.6 STDR (Yang, Chen, Wang, Li, Pan, Cheng, Liu; Tsinghua Shenzhen / Zerith; 2606.31377)

### Method

Dense reward for RL from scratch, built from unstructured expert videos. **Stage segmentation**: a VLM (Qwen3) given the task description and uniformly sampled frames returns JSON segments; K per task is the robust mode across demos; VLM output is used only offline as weak labels. **Stage discriminator**: ResNet features over a window of L frames → LSTM → k_t ∈ {0, …, K−1}; a VAE on the window gives an OOD reset (reconstruction error > δ → k_t ← 0); an MLP **grasping-regulation** gate lets k_t advance past the grasp stage only when it predicts a stable grasp. **Progress estimator**: ResNet + FiLM on the stage embedding → p_t; losses (eqs. 2–6): pairwise ranking with margin, isotonic (no decrease across consecutive steps), visual consistency (strong vs weak augmentation), regression to within-stage t/T; video rewind as negatives. **OOD-aware fusion** (eqs. 7–8): Mahalanobis distance d_ood on the feature layer; p = g(d_ood)·p_reg + (1−g)·p_near with a retrieval-based nearest-expert estimate p_near, and p = 0 beyond η. Reward (eq. 10): 1.0 on success, else k_t/K + p/K.

### Protocol and numbers

- 14 tasks: MetaWorld 8, ManiSkill 3, Franka Kitchen 3; **30 expert demos per task**; PPO; **4 seeds, mean ± std curves** (Fig. 2). Baselines: sparse, Human-Dense (official shaped reward from ground-truth state), VIP, LIV (fine-tuned by TCL on the same demos for the real-robot comparison).
- Table I (ManiSkill, best checkpoint, 50 episodes): PullCube 0.959 ± 0.029 vs Human-Dense 0.955, DP 0.120, LIV 0.051, VIP 0.000; PushCube 0.943 vs 0.931; PickCube 0.632 vs Human-Dense 0.961 (the one loss).
- Ablations (Fig. 4): stage-only converges late with high variance; progress-only plateaus lower (loses the stage order); Fig. 5: without the grasp gate, "stage leakage" — the reward advances before the object is held, and the agent moves with an empty gripper.
- Real (§IV-B): Franka + RealSense, 15 success videos per task (10 train / 5 eval) + some failure videos; Fig. 3: STDR monotone on successes and low on failures (drop at the moment the toy is dropped); LIV/VIP plateau on successes and stay high on failures. **No RL on the real robot.**

### What is not there

- Real-robot numbers: Fig. 3 is qualitative (curves), no AUROC, no counts.
- The VLM segmentation's agreement with human stage boundaries is not measured.
- A learned-from-length or elapsed-time baseline for the progress signal.
- Only consistent-stage-order tasks (stated limitation).

### For the SO-101

Not an RL-on-the-Orin paper; the usable idea is the **grasp gate**: a stage cannot advance until a separate verifier says the object is held. That is the v1.5 tier structure applied inside a reward, and it is the specific defence against the "stage leakage" you saw when the post-grasp VLM checkpoint fired before the lift (41/42 flagged). Second usable idea: the isotonic + rank + rewind loss set is a complete recipe for a progress head on your 69 rollouts that is not just elapsed time, and the honest test is partial correlation with length (chapter 13.3.2).

---

## 13a.7 FailSafe (Lin, Duan, Fang, Fox, Krishna, Tan, Wen; NTU / AI2 / UW; 2510.01642)

### Method

A data pipeline in ManiSkill: (I) **failure generation** — at a motion-planner stage boundary, perturb the pose by translation (±0.1), rotation (±1 rad) or no-op, and record the rollout if the task then fails; (II) **action collection** — for a deviated pose P_d (searched from step 10 onward) and a corrective pose P_c on the correct trajectory (window: 10 steps after start to 3 before end), ΔA = 7-DoF difference; (III) **systematic verification** — replay A → P_d → P_c → B → C → D and keep the pair only if the task now succeeds. 131k failure–action pairs + 56k clean trajectories over pick / push / stack cube (Table I, 2.3 : 1 failure : success). Fine-tune LLaVA-OneVision-7B (full, one epoch, 32 H100) on a VQA template: given 10 consecutive frames from three cameras and the task, output sub-task, yes/no, failure type, ΔA. Deployment: every 10 steps FailSafe-VLM looks at a window; if "yes", its ΔA is executed, then the base VLA resumes.

### Protocol and numbers

- Franka in ManiSkill; π0-FAST, OpenVLA, OpenVLA-OFT fine-tuned on 1,000 clean trajectories per task; test seeds with new spatial configurations; **the FailSafe-VLM sees a camera view it was never trained on** (the VLA's training view).
- Table II (75 runs per cell, Table VI): π0-FAST 78.7 → 82.7 (+4.0); OpenVLA 14.7 → 37.3 (+22.6); OpenVLA-OFT 90.7 → 98.7 (+8.0). Table III novel objects (sphere, charger) with OFT: 53.3 → 70.7 (+17.4). Table IV novel embodiment (xArm 6, VLM never trained on it): 85.3 → 92.0 (stack cube 56 → 76).
- Table V (1,712 held-out entries, 20 seeds): binary success FailSafe-VLM 0.909 vs GPT-4o 0.70, Gemini-2.5-flash 0.62, Qwen2.5-VL 0.24; failure-type accuracy 0.837 vs < 0.2; ΔA cosine 0.65 vs ≈ 0 (Qwen always outputs "no failure" and a zero action).
- Overhead: +3.8 to +9.1 s per episode, mostly simulator replanning.

### What is not there

- Real robot: none ("prohibitively difficult to obtain through real-world data collection" is the motivation, not a result).
- Object-level errors (stated limitation): only motion-level deviations are injected and corrected.
- A baseline that intervenes with a *fixed* retreat (Spotter's "fixed retry") to separate detection from correction.
- Intervals on the 75-run cells.

### For the SO-101

You cannot inject failures with a motion planner, but you can with the **leader arm**: a teleop deviation at a stage boundary, recorded with the recovery that follows, is this paper's data pipeline on your hardware, and the tidy-loop "deliberate corrections" demos in chapter 13.3.1 are already half of it. The honest expectation: their corrections are 7-DoF Cartesian deltas on a Franka; yours would be joint-space, and the verification step (does the replay succeed?) is a robot-day cost per pair. The transferable method result is Table V: off-the-shelf VLMs reach ~0.6–0.7 on "is this failing" and **~0 on "what to do about it"**; use them as judges, not correctors.

---

## 13a.8 VLA-Corrector (Song, Qian, Feng, Song; 2609.06508)

### Method

A frozen π0.5 (OpenPI `pi05_libero`, replan every 5 of the chunk) plus a **Learned Verifier** on observable history only: two 112 × 112 views → parameter-free spatial-statistics features (3×4×4 pooled RGB + mean + std = 54 per view) → Linear 64; [state 8-d; action 7-d] → MLP 32; GRU(64) over an 8-step window sampled every 2 env steps; two heads: binary risk p_t and stage ĉ_t ∈ {approach, align, grasp, move, place}. **Teacher**: a privileged rule-based state machine on simulator state produces stage labels and failure-event steps; f_t = 1 within 10 steps of an event, 0 beyond 20, the band in between excluded (eq. 7). Gate: τ = 0.40, four consecutive positive windows, 10-step cooldown; stage routes a recovery prompt fragment r_t appended to the instruction (q ⊕ r_t); low stage confidence (< 0.9625) → generic prompt; up to 100 recovery actions, then the original instruction resumes. Verifier: 120 episode-disjoint episodes, 60/20/20, 30 epochs, CPU; 0.457 ms p50 inference.

### Protocol and numbers

- LIBERO (5 suites) and LIBERO-Plus difficulty 4 and 5; three configurations: No Assistance, Rule + Prompt (privileged verifier, an **efficacy upper bound**), Learned Binary + Prompt. Perturbations: action noise σ = 0.1, target-object shift 0.1 m, goal-region shift 0.1 m, combined; **1,000 episodes per condition** (Table 3).
- Fig. 3: standard LIBERO 70.9 → 82.6 (rule) / 80.2 (learned); Plus-4 73.7 → 79.7 / 79.0; Plus-5 66.7 → 73.6 / 72.6. Table 3 combined perturbation 70.7 → 82.6 / 80.2. The learned verifier is within 2.4 points of the privileged one.
- Verifier quality: accuracy 93.1 %, precision 61.1 %, recall 94.9 %, F1 74.4, AUROC 0.902, AUPRC 0.712; trigger precision 79.2 % (validation) / 75.0 % (test replay); recovery success 77 % (rule) / 75 % (learned). Table 4 ablations: single frame F1 −15.0; agent view only −10.8; vision only −6.3; state-action only −15.8.
- Real: the same frozen policy on a physical arm, three tasks, **40 trials each**, perturbation = moving the target after execution begins; 82.5 → 88.3 average (Pick-and-Open 90 → 92.5, Place 90 → 95, Pour Water 67.5 → 77.5).

### What is not there

- Intervals: 1,000 episodes in sim make ±3 points resolvable; 40 real trials do not resolve +2.5 or +5 (Wilson on 90 % at n = 40: 77–96 %).
- The real robot is not named; the verifier was trained on simulator teacher labels and "no real-world data are used for policy fine-tuning or verifier training" — the real result is zero-shot from sim features.
- A baseline that triggers recovery at random or on a timer, to separate "prompting recovery helps" from "the verifier chooses when".

### For the SO-101

Two things. First, the **precision/recall split** of a 61 % / 95 % verifier shows what a usable gate looks like when a false trigger is cheap (a recovery prompt costs 100 steps, not a trial): tune for recall, confirm over four windows, cool down. Your monitor bar (≤ 1/20 false interventions) assumes a false trigger wastes the trial; if the intervention is instead "re-observe and re-prompt", the bar can relax and a 6/8-at-7/42 head may already be a useful gate. Second, the 54-feature parameter-free visual front end plus a GRU reaches AUROC 0.90 on LIBERO; it costs nothing on the Orin and is the right *floor* baseline to put under any W1-based monitor.

---

## 13a.9 Spotter (Li, Zhao, Yang, Xu, Wang, Liew, Qu, Shen, Pan; 2609.36808)

### Method, and the diagnosis first

§3.1 is the experiment to remember. With an **oracle** marking the failure and three retries allowed: resampling with three seeds corrects 7.5 % (Cosmos Policy) / 8.75 % (π0.5) of failed grasps; VLM language feedback 9.69 %; recovery fine-tuning 8.13 / 8.44 % (only ~3 % of rollouts yield usable recovery data). Judging a correction (Fig. 2b, 320 failed grasps, eight restarts × 32 steps, pairwise ranking of successful vs failed candidates): Cosmos value head 0.66 at normal states → **0.57 after a failure**; Consistency-Consensus 0.45 (below chance) post-failure; GeoBoN, RCS ≈ 0.5 throughout; a VLM given only images around the saved state 0.53 (GPT) / 0.51 (Qwen). So neither the policy nor its scorers can repair a known error, because both learned from successes only; repair needs the episode history and text.

Spotter: the policy leads and runs continuously; a locally served **screener** (Qwen3.8-27B) scores each completed chunk's record z_i and flags when r_i ≥ τ (tuned for recall) or on empty-gripper evidence; after two consecutive flags the **judge** (Qwen or GPT-6 Astra) is called with the episode history and returns (verdict, diagnosis, expectation, plan of primitives) — move to point (pixel → 3-D via depth, which the VLM never sees), move by offset, lift, rotate wrist, set gripper, retreat to a recorded earlier pose, hand back. The policy may run at most K chunks ahead of the last check (eq. 1); it pauses only on a confirmed error or at the lead limit. Each intervention states an **expectation** for the next window; if it fails, the next repair must be of a different kind. Budgets on segments and interventions; every motion counts toward T. A library of general lessons is learned in sim with resets; at test time nothing is reset.

### Protocol and numbers

- RoboCasa, 24 tasks × 50 episodes, **fixed seeds, every episode with and without Spotter under the same scene / instruction / step budget**; repairs consume the policy's budget. Cosmos Policy and π0.5; Harness VLA (VLM-led, 40 rounds, 5,000 steps, with/without privileged success predicate) as the comparison.
- Table 4: Cosmos 66.7 (1.0× budget) / 67.9 (1.8×) → Spotter (Qwen, full) 71.7 (+3.8) / (GPT, 1-shot) 73.5 (+5.6); π0.5 62.5 / 64.3 → 68.4 (+4.1) / 71.8 (+7.5); gains concentrated on pick-and-place (+10.3 / +9.0). **Harness VLA without privileged info loses: 65.2 (−2.7) / 60.5 (−3.8).** Screener + fixed retry (retreat, no judge): 66.9 / 62.1 — detection alone is not enough. Oracle + fixed retry 67.1 / 63.1 — even perfect detection with a dumb retry does nothing.
- RoboTwin 2.0 Hard: π0.5 47.2 → 51.2 (Qwen) / 57.0 (GPT).
- Real Franka Research 3, three tasks, **10 trials each**: 16/30 → 25/30; long-horizon 3/10 → 7/10.
- Cost (Fig. 6, Table 5): per successful episode +13 s (Cosmos) / +16 s (π0.5) over the policy alone; Harness VLA 146 / 164 s vs 34 / 23 s alone; judge calls on successes 0.6–1.1 per episode, on failures ~10; repairs on failures 2.4–3.7.

### What is not there

- Real-robot n = 10 per task, no intervals; the headline is simulation at 1,200 episodes per configuration, which is enough for the +4–7 points.
- The screener's own FP / FN are not tabulated in the main text.
- A world-model-in-the-loop variant (their stated next step for irreversible damage).

### For the SO-101

This is the v1.4/v1.5 harness measured, and three of its numbers go straight into your spec. (1) **VLM-led loses** (−2.7 / −3.8) unless the VLM sees the success predicate; your harness design (agent dispatches, verifier reads at rest, policy runs continuously) is on the right side of that line and you can now cite why. (2) **Detection without a repair plan is worth nothing** (oracle + fixed retry: +0.4); so a monitor gate on the Orin is only as valuable as the recovery primitive it triggers — retreat-to-recorded-pose is the one you can build this month. (3) The §3.1 result that best-of-N scorers fall to chance after a failure is the mechanism behind your "judge only at rest" rule and the reason the harness never lets the policy grade its own retry. The expectation-then-verify loop (state what should happen in the next window; check it; if not, try a different kind of repair) is the per-intervention version of pre-registration, and it belongs in the harness as a rule.

---

## 13a.10 No Free Checker (Wan, Yue, Liu, Chu et al.; Zhejiang / CityU HK; 2609.09250)

### What it is

A survey of ~150 verifiers (verifier : (context, candidate) ↦ score) on two axes: **availability** (cost, earliness, density) and **credibility** (how much a high score says about the task; falls as the judgment becomes gameable). Four families by who supplies the judgment: human (compare, score, intervene, validate generated rollouts), rule-based and formal (STL satisfaction margin with or without conformal calibration; geometry such as RINSE's spectral arc length; LLM-written rules; goal predicates; CBF certificates; reachability; symbolic feasibility), learned and pretrained (trained vs training-free), model-intrinsic (the policy's own uncertainty; prediction-based). Table 1 crosses "what the score asserts" (success, progress, safety, execution quality, preference, failure/uncertainty, training value of a demo) with the source.

### The part to memorise: §6, validating the verifier

- **Table 5**: four uses × how candidates arise × requirement × cost of error (quoted at the top of this chapter). Curation and ranking select among non-adversarial candidates, so average accuracy split by error type suffices; training reward and runtime gating expose the score to a search, so "no exploitable region" is required and one rare wrong-scoring region makes the score unusable.
- **§6.1 agreement on a fixed set omits three things**: how a disagreement splits (verifier wrong vs reference wrong; Lee et al. find frontier VLMs call a failed drawer-opening a success); the **confidence interval** (blind randomised trial budgets, time-to-success distributions, sequential designs, the interval estimators for RL comparisons on few seeds); and **the kind of rollout** (GE-Sim 2.0 scores one verifier on world-model-generated and on recorded video of the same tasks and finds an eight-point gap; SureSim pairs few real trials with sim; OpenGVL measures across embodiments). Table 6 audits 13 resources; WorldEval and WorldGym never measured verifier error on the deployment distribution.
- **§6.2**: the gain in the trained policy is what matters and is rarely reported with the agreement rate for the same verifier; datamodels and self-improvement loops are the two methods that do compute it.
- **§6.3 under optimisation**: reward hacking is well studied for language (scaling laws of over-optimisation, token-space attacks) and "largely undeveloped" in robotics; a **held-out verifier panel** protocol (score rollouts with the training verifier and with verifiers the policy never saw; report the fraction the panel rejects among those the training verifier passed); conformal margins are invalidated by filtering on the verifier's own scores or by training against it (exchangeability breaks), so a guarantee holds for a frozen policy and weakens after retraining.
- **§6.4 nine metrics** and "proxy gain that transfers": Δ_panel / Δ_proxy between first and last checkpoint, with both differences reported, 1 = all real, 0 = none, negative = the panel scores the policy lower while the proxy rises.

### For the SO-101

This is the vocabulary your verifier tiers (v1.5 §1.1) were reaching for, and three of its rules sharpen them. (1) **Say which use** every judge serves (G9c-3: curation + ranking; a monitor: runtime gate) and apply that row's requirement — your zero-FP rule for `demos/` is exactly "a false positive trains on failure". (2) **Report disagreement direction and the interval**: your report cards have FP and FN; add the CI (Wilson) and the kind of rollout (which policy, which session) the rate was measured on, because coverage shift is the "kind of rollout" problem. (3) If a learned judge ever selects training data in a loop (the flywheel), freeze a **held-out panel** (human labels on a fixed set + G9c-3) and report the proxy-gain ratio at every retrain; without it the flywheel is unverifiable by construction. Also take the survey's one-line test for any verifier paper: does it measure its own error on the distribution it is deployed on? Most do not.

---

## 13a.10a PRIME: progress from an experience graph (Tsinghua BNRist; 2610.23769; run 2026-10-29, void)

### Method

Build a graph over demonstration states: nodes every few frames, temporal edges along each demonstration, cross edges between demonstrations where the TCP and gripper are close **and** a visual gate (DINOv3 features) agrees; weight edges by TCP + gripper distance. Progress of any state is its shortest-path (Dijkstra) distance to the success set. The paper uses it to weight training samples: π0.5 on charger insertion 10 % → 50 %; local-precision accuracy 92.5 % vs 70.0 % for the best baseline. Piper-X, 20–50 demonstrations per task, 30 trials per policy.

### The translation and the bar

Only the estimator was tested, as a failure monitor on the standing REAP bar: the 90 "blue bowl in pink bowl" teleop episodes as the graph, DINOv2-S as the gate, the 50 W1-A episodes as queries. Pass: ≥ 5/8 failures at ≤ 2/42 false interventions **and** at least one more catch than the same graph with the visual gate off (the kinematic-only graph, which the paper never ran). Protocol frozen 2026-10-23. Before any rollout number: a measurement check that the graph is a progress estimator at all — **held-out teleop monotonicity**, median Spearman of graph distance against time on 10 held-out teleop episodes, ≥ 0.8.

### What happened

The reference repository contains a README and figures but no code, so parity fell back to the protocol's 3-episode hand example (pass). Parts summed (10,564 nodes = expected; 10,474 temporal edges = expected). Thresholds from the data: ε_x 7.2 mm, ε_y 8.5 mm, ε_z 2.8 mm. **Held-out monotonicity 0.51 < 0.8.** By the protocol the graph is broken and no rollout number is read. The script had computed rollout scores; they are not reported anywhere. Off-graph rate on success queries: 59 %.

### Reading

With thresholds set from 10 Hz teleop steps (millimetres), a held-out demonstration either falls off the graph or joins it through a few cross edges, so its distance to the goal is not monotone in time. Fixing that means choosing thresholds or the frame rate after seeing data, which the protocol forbids; a new protocol would have to fix them first. Verdict **void**: not a fail of PRIME, a fail of this instantiation's instrument, caught before it could produce a misleading number.

### For the SO-101

The case belongs to chapter 6 more than to this chapter: a check that can void the run before the result exists, written into the protocol, and honoured even though the result had already been computed. It is also the third independent measurement (with G6 and SURVEYOR gate 0) that distance in a learned or engineered state space is not task progress on this rig at this data scale.

## 13a.10b RoboFQE (University of Wisconsin–Madison, CoRL 2026; 2610.18699) — read, not tested

### Claim

Estimate a policy's success rate **without running it**: fitted Q-evaluation on DINOv3 features over offline trajectories only, with terminal rewards from a learned success detector. On an SO-101 with top and wrist cameras — this rig's setup — it estimates SmolVLA, π0.5 and a multitask BC policy (ground truth 58.5 %, 81.0 %, 33.0 % from more than 100 episodes per task) with 7.5 ± 5.0 % absolute relative error, Pearson r = +0.90; world-model evaluators score 10.3–24.5 %. LIBERO-Long: 5.63 % vs over 50 %.

### Why not tested, now

The setup matches; the ground truth does not. Four of the five policy families here sit at 8–9/10 on 10-trial gates, and the 95 % interval of 9/10 runs from about 0.55 to 0.99: any estimate within ±0.3 would pass and none could be ranked. The one policy that separates (TurboVLA, ~2/21) is already caught for free by the task-swap gate. The test becomes informative only when two checkpoints have ≥ 30 sealed-layout trials each and differ by ≥ 15 points. The paper also skipped the trivial comparator: held-out teleop action error (the validation loss) as a ranking.

### What it changes

NEXT_ANGLES §11 (active data selection) will produce exactly that ground truth: two fine-tunes from one base, 30 sealed-layout trials each. Before that evaluation day, both are scored with RoboFQE (DINOv2-S features, terminal reward from the G-flora judge, neither checkpoint's own rollouts in the offline data) and with held-out action MSE; both predicted success rates are frozen in the protocol. Afterwards, each one's absolute error and whether it ordered the pair are reported, not gated: one correct order is one bit.

### The general point

This is chapter 7's power argument used to decide *not* to run a test. A 10-trial gate cannot distinguish an estimator that is right from one that is off by 30 points; the right response is to say when the test becomes informative and attach it to the experiment that will produce that ground truth anyway.

## 13a.11 How this family evaluates, side by side

| Paper | Needs failure labels | Platform | Real n | Seeds / intervals | Matched control | Trivial baseline |
|---|---|---|---|---|---|---|
| FAIL-Detect | no | robomimic + Franka | low (Table III) | none | post-hoc vs learned scores | none |
| SAFE | yes (trajectory) | LIBERO, SimplerEnv, Franka, WidowX | 30+30/task; 532 | 3 random splits | 14 adapted baselines | none |
| ReAP | yes (ramp) | LIBERO, RoboTwin, real π0.5 | app. C.4 | 3 seeds ± std, 5 α | random detector; shuffled chunk | none |
| FARM | yes | LIBERO + 4 real incl. **SO-101** | 497 / 100 / 70 / 778 | 5-fold OOF, 3 seeds | low-order statistics; block-6 | common-horizon truncation |
| Robometer | preference pairs | 6 embodiments OOD; DROID RL | 20 trials; 100 traj | none | RoboReward same data | none (**not length**) |
| STDR | no (stages from VLM) | MetaWorld, ManiSkill, Kitchen; Franka videos | qualitative | 4 seeds ± std | Human-Dense (oracle reward) | sparse |
| FailSafe | synthetic | ManiSkill | none | 75 runs/cell | three VLMs same prompt | none |
| VLA-Corrector | teacher (sim) | LIBERO(+Plus), real arm | 40/task | 1,000 eps/condition | privileged rule verifier (upper bound) | none |
| Spotter | no | RoboCasa, RoboTwin, Franka | 10/task | fixed seeds, paired | oracle + fixed retry; screener + fixed retry | policy alone at 1.8× budget |
| No Free Checker | — | survey | — | — | — | — |

Read with chapters 3–7:

- **Nobody runs episode length or policy identity.** It is the baseline that beat Robometer on four of four held-out tasks and it does not appear in any of the nine experimental papers. That is the one sentence from this chapter to say in an interview, with the G9b numbers.
- The two cleanest designs are the ones with an **oracle or upper bound**: VLA-Corrector's privileged verifier (learned within 2.4 points) and Spotter's oracle-plus-fixed-retry (+0.4: detection alone is worthless). FARM's common-horizon truncation and ReAP's shuffled-chunk control are the two cheapest **shortcut-removing** controls and both are missing from your register.
- **Onset labels are avoidable**: ReAP's ramp target and FARM's partial-history curve both report earliness without them. Your REAP v2 (2026-10-23) used exactly this: no onset labels, ramp target, common-horizon cut; the two confounds left and the verdict became a clean fail (logistic 3/8 at 5/42, AUROC 0.789; true vs shuffled +0.08, so action-conditioned). The 25 hand labels were never needed.
- The monitors that work on hardware (SAFE, ReAP, FARM-Adapt, VLA-Corrector) all share one property: a **supervised probe on a fixed representation**, trained on 50–500 labelled episodes from the deployed policy. The unsupervised ones (FAIL-Detect bands, W1 surprise) fail the same way on hardware: the success distribution shifts with the setting. That is a field-level finding and your W1-A/C is one more data point for it.
- Real-robot n is 10–70 per task across the chapter; the survey (§6.1) names the missing interval as a field-wide omission. Your episode bootstrap is above the norm; say so once, with the Wilson numbers next to theirs.

## 13a.12 Corrections and additions to chapter 13 and the register

- 13.2 monitors: the REAP-style head is renamed in the register to what it is ("logistic probe on W1 latent + actions"); the next version adopts ReAP's **ramp target** (removes the onset confound), the **shuffled-chunk control** (proves action use), **bTWA** as the reported metric beside your 5/8-at-1/20 bar, and a **VLA-Corrector-style parameter-free GRU** as the floor baseline.
- 13.1.2 learned judges: add the instruction-negative test on your own frames before any reward-model gate; add FARM's SO-101 row (0.58–0.70 zero-shot, 0.83 adapted on ~47 labels) as the external prior for a frozen-world-model readout on this arm.
- 13.1.1 tiers: name the **use** (Table 5) on every verifier card; add Wilson CI and "kind of rollout" (policy, session) to each report card; freeze a held-out panel before the flywheel ever selects training data, and report Δ_panel / Δ_proxy.
- Harness (v1.5): adopt Spotter's three rules — policy runs continuously and the agent never leads; a detection without a recovery primitive is not a gate; every intervention states an expectation that the next window must satisfy or the next repair must differ. Cite Table 4's −2.7 / −3.8 for VLM-led.
- 13.3 data: the leader-arm deviation-then-recovery recording is FailSafe's pipeline on your hardware; log it as a candidate session type, with the verification cost (replay success) named.
- Chapter 6 instrument: add FARM's common-horizon truncation as a standing shortcut check for any judge trained on your episodes.

- 13.2 monitors, the count to quote: seven latent-space monitors have now failed the standing bar on this arm (W1-A/B, W1-C1, W1-C2, G15, G16-B, the REAP head, SURVEYOR's accept rule, 12b.8c); the PRIME graph (13a.10a) never reached the bar because its own check voided it. Verify at rest remains the only judge that passes.

## 13a.13 Exercises

1. Retrain the REAP head with the ramp target and no onset feature; evaluate with (a) true chunks, (b) chunks shuffled from other episodes of the same task, (c) zero actions. Report bTWA and your bar for all three. If (a) ≈ (b), write the correction row.
2. Implement SAFE-MLP on SmolVLA's action-expert input features in vla.cpp (export the vector), with the functional CP band on the 42 successes; three random seen/unseen splits over your tasks; ROC-AUC on the max-so-far score; episode length beside it.
3. Compute FARM's common-horizon truncation for the bowl and cube tasks; re-run G9c-3's report card and the REAP head on the truncated episodes; say what changed.
4. Run Robometer's instruction-negative test on 20 of your episodes (same frames, swapped instruction); report the score drop; decide whether the model reads the instruction on your data.
5. Write the Spotter expectation-then-verify rule into the harness spec as a budgeted loop, and the retreat-to-recorded-pose primitive as the first recovery; pre-register the robot-day block that tests "detection + retreat" against "detection alone" with paired layouts.
6. Build the verifier card template from No Free Checker §6.4: use, FP, FN, Wilson CI, kind of rollout, disagreement direction, latency, and (for anything in a loop) the proxy-gain ratio. Fill it for G9c-3 and Flora from existing data.
