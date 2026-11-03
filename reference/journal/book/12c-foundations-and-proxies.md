# Chapter 12c. The foundations, and why the loss is not the result

Five papers added to the knowledge base on 2026-10-23 because the book cited them without holding them. Two are foundations (LeJEPA, which is where SIGReg comes from, and V-JEPA 2, which is where the action-conditioned planner every later paper builds on comes from). Three are the strongest statements in the field of the thing chapter 3 keeps saying: the number you train on is not the number you want. Each is read as in 12a: objective, protocol with n and seeds, numbers with table references, what was not run, what it means on this hardware.

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | LeJEPA | 2511.08544 | Isotropic Gaussian is the provably optimal embedding law; SIGReg enforces it with one hyper-parameter; no EMA, no stop-gradient |
| 2 | V-JEPA 2 | 2506.09985 | 1M hours of video → ViT-g; V-JEPA 2-AC: 300M predictor on 62 h of DROID; zero-shot Franka pick-and-place, 10 trials/task |
| 3 | One-step fidelity ≠ planning | 2610.19375 | Frozen-decoder protocol; r = −0.010 across 15 checkpoints; matched continued-training control absorbs a "grounding" gain |
| 4 | Decision-relevant prediction error | 2609.32322 | Iso-error families: 1 % total-error difference → 60-point planning difference; DRPE ρ = −0.98 vs total ρ = −0.32 |
| 5 | SALT | 2609.33595 | State-affine transition + rollout training: 1.5–2.2× *worse* one-step error, +10 pp planning; mismatch failures 23 → 2 % |

---

## 12c.1 LeJEPA (Balestriero, LeCun; 2511.08544)

### What it claims

Two axioms for a JEPA: solve the prediction task, and make the embedding distribution isotropic Gaussian. The second is not a heuristic: §3 derives that the isotropic Gaussian uniquely minimises downstream prediction risk over families of linear (§3.1) and non-linear (§3.2) probes. So the regulariser is a *target*, and the question becomes how to enforce a high-dimensional distribution match cheaply.

### SIGReg

Sketched Isotropic Gaussian Regularisation. Project the batch of embeddings onto M random unit directions (Cramér–Wold: if every 1-d projection is N(0,1), the joint is isotropic Gaussian); on each projection compute the **Epps–Pulley** statistic — the weighted ℓ2 distance between the empirical characteristic function and e^{−t²/2}, integrated by a 17-knot trapezoid on [−5, 5]. Why that test and not a moment or CDF test (§4.2): moments explode gradients at high order; CDF tests need sorting, which is non-differentiable and breaks DDP; the characteristic function is bounded, its gradient is bounded by 4σ²/N (Theorem 4), and the whole thing is O(N) with an all-reduce of one tensor (Algorithm 1, ~20 lines). Mini-batch bias is O(1/N) (Theorem 6) and "not a concern even at batch 16".

Objective: L = (λ/V) Σ_v SIGReg(z_v) + ((1−λ)/B) Σ_n ‖μ_n − z_{n,v}‖² with μ_n the mean of the global views. One hyper-parameter; no EMA, no stop-gradient, no prototypes, no schedulers. Recommended λ = 0.05, V_g = 2, V_l = 8, batch ≥ 128.

### Protocol and numbers

10+ datasets, 60+ architectures. ImageNet-1k linear probe, frozen backbone: ViT-H/14 79 %; ViT-L 77.1, ConvNeXtV2-H 78.5. Stability: λ sweep on ImageNet-100 ResNet-50 flat over a decade (Fig. 8); slice count, integration domain and quadrature knots have "negligible impact" (Table 1); batch 128 works. Transfer (Table 2, frozen, 8 datasets): LeJEPA ViT-L at 100 epochs 79.5 avg vs I-JEPA ViT-H at 300 epochs 78.5. In-domain (Galaxy10, 11k images, Fig. 12 / Table 3): small LeJEPA backbones beat DINOv2/v3 transfer at every shot count. And one result for chapter 6: **the training loss correlates with downstream linear-probe accuracy at Spearman 0.945** on ImageNet ViT-B (Fig. 1 top-left) — "the first practical loss for model selection without supervised probing."

### What is not there

- No world model, no actions, no planning: this is representation learning on images. LeWM adds the predictor.
- The loss-tracks-accuracy result is for *classification* probes, which is exactly what §3 optimises for; 2610.19375 shows it does not carry to planning.

### For the SO-101

This is the source of the 12.2 explanation, and it corrects it: SIGReg is not "a VICReg-like variance term". It is a distribution test with a derivation, and its properties (bounded gradients, no sorting, O(N), DDP-friendly) are the reason W1 trains stably at batch 32 on a rented 3090 with no EMA. The one line to keep for interviews: *collapse is prevented by matching the embedding law to a target that is provably optimal for linear readout, not by a trick.* And the warning: the Spearman-0.945 loss-vs-probe result is why people select JEPA checkpoints by loss, and 12c.3 is why you must not.

---

## 12c.2 V-JEPA 2 (Assran, Bardes, Fan, Garrido et al., Meta FAIR; 2506.09985)

### Method

Stage 1: self-supervised video pretraining scaled to >1M hours (22M videos), ViT-g (1B), masked latent prediction with an EMA target, 3D-RoPE, progressive resolution. Stage 2, **V-JEPA 2-AC**: freeze the encoder; train a ~300M **block-causal** predictor (24 layers, 16 heads, 1024-d) that takes each frame's patch features interleaved with an action token and an end-effector-state token (3D-RoPE on patches, temporal RoPE only on action/state tokens) and predicts the next frame's features. Loss: **teacher-forced L1 + one-step rollout L1** (Fig. 6), on **DROID raw** (23k trajectories, ≈62 h, successes and failures, left camera only, 4 fps, 256 px). Planning (eq. 5): energy E = ‖P(â_{1:T}; s_k, z_k) − z_g‖₁, CEM over T-step action sequences (Cartesian Δ + gripper, norm clipped to 0.1), execute the first, replan; ~16 s per step.

### Protocol

Two Franka arms with Robotiq grippers in two labs, zero-shot (no fine-tune on either lab). Image goals; for pick-and-place three sub-goal images (grasped; near target; placed) switched on a fixed schedule (4 / 10 / 4 steps). Baselines: Octo (OXE-pretrained, fine-tuned on DROID with hindsight goal relabelling) and action-conditioned Cosmos-7B fine-tuned on DROID. **10 trials per task per lab**, permuted object positions and start poses.

Table 2 (avg of two labs): Reach 100 / 100 (Octo / V-JEPA 2-AC); Grasp cup 15 / 65, box 0 / 25; Reach-with-object cup 15 / 75, box 70 / 75; Pick-&-place cup 15 / 80, box 10 / 65. Fig. 9: the energy landscape over (Δx, Δy) for a reach is smooth and locally convex with the minimum near the true action. Limitations (§4.3): camera position had to be hand-tuned ("we manually tried different camera positions"); long-horizon planning limited by error accumulation; the model has no notion of the action coordinate frame when the base is out of view.

### What is not there

- n = 10 per cell. 65 % is 6.5/10 averaged over labs; the Wilson interval on 13/20 is 41–81 %.
- No seeds; no intervals; a single predictor.
- **A bug.** The JEPA-WM study (2512.24497, App.) reports that the released code miscomputes the 2-step rollout loss (it trains the model to reproduce its own prediction), and that the public checkpoint scores much lower on their DROID action-score protocol than their bug-fixed retrain — while decoding visually plausible rollouts and passing a counterfactual check. Read that twice: a checkpoint can look right in every visualisation and be wrong on the number that matters.

### For the SO-101

This is the ancestor of every planner in 12a/12b, and three facts transfer. (1) The two-term loss (teacher-forced + rollout) is the original source of RoboJEPA's AR term and SALT's rollout training; W1 has only the first. (2) The camera-position sensitivity in §4.3 is your SEVO finding (14a.7) from the other side: a world model trained on one camera learns that camera's action frame; moving the mount is a distribution shift the paper could not fix. (3) The bug report is a chapter-6 case: a parity check against the paper's *numbers* (their DROID action score) caught what a parity check against *visualisations* would have passed. When you reproduce anything from this family, compare to a number, not a picture.

---

## 12c.3 One-Step Prediction Fidelity Is a Poor Proxy for Planning (Cheung, He; 2610.19375)

### The protocol (§3), which is the contribution

Fit a decoder D from encoder embeddings of **real** observations to physical state (angles as sin/cos; error normalised per component by training variance) on held-out training episodes; freeze it; apply it to the predictor's output on disjoint test episodes. Three numbers: E_repr = d(D(enc(o_{t+1})), s_{t+1}) (what the encoder holds); E_pred = d(D(ẑ_{t+1}), s_{t+1}) (what the predictor produces); **E_excess = E_pred − E_repr**, which is exactly zero for a perfect predictor whatever D does, and which flags when the protocol has no power (E_repr ≈ E_pred means the error is upstream of the predictor). Decoder-free check: latent drift ‖ẑ − z‖, correlated with E_excess at r = 0.85 across the ladder.

Intervention: warm-start LeWM's released checkpoints, continue 5 epochs at lr 1e-5, batch 128, **three training seeds**; arms = *continued* (same two-term loss) and *state-grounded* (+ an auxiliary head predicting physical state from encoder embeddings, weight 0.1). Planning: the released CEM/MPC evaluation unchanged, 50 goal-conditioned trajectories, **three paired planning seeds at every training seed**, 10,000-rep hierarchical paired bootstrap. Stated limits: three seeds give poorly calibrated top-level intervals; 50 trajectories put a 4–7 pp floor on resolvable effects.

### Results

- Non-degeneracy first (§5): shuffling the action raises E_pred 29× (PushT), 16× (TwoRoom), 2.1× (OGBench-Cube); persistence/constant-velocity skill 0.87+ on the first two; **OGBench-Cube's predictor is worse than constant velocity (skill −0.74) and E_excess is 0.3 % of E_pred** — the protocol detects, without a downstream task, that there is nothing to measure there.
- Within PushT (Table 1): epoch 0 → 1 moves both (fidelity −32 %, planning +4.4 pp); epochs 1 → 5 move fidelity −11 % more, monotone in every seed, and planning +0.22 pp with per-seed changes +2.0 / 0.0 / −1.3. **Correlation over the 15 post-epoch-1 checkpoints: r = −0.010** (drift: −0.175). Including epoch 0 gives r = −0.88, carried entirely by one step.
- Not a ceiling (§6.1): stratify by a model-independent data-support measure (8×8 grid of block positions); low-support planning starts at 80 %, peaks at epoch 1 (+9.2 pp), then declines to +7.5 while fidelity keeps improving. Where data are sparse the model both predicts worse (block-position error contrast −0.0021 [−0.0024, −0.0019]) and plans worse (18–20 pp gap).
- Across environments (Table 2): TwoRoom −93 % fidelity → −0.7 pp planning; PushT −40 % → +4.9 [+1.1, +8.9], the only resolved effect, and it belongs to the *control* arm.
- The two warnings (§8): untouched baselines vary **by up to 26 pp across planning seeds** (OGBench-Cube sd 13.1; PushT sd 6.0) — larger than any induced effect; and **state grounding is indistinguishable from five more epochs of plain training** on every measure (paired differences all contain zero). Reported without the matched arm, "an auxiliary grounding loss cuts physical error 90 %" would have been true and misattributed.

### What is not there

- One architecture (LeWM, 15M), three low-dimensional sims, one planner, one checkpoint family; the within-task ladder is one environment.
- No mechanism claimed ("we advance no mechanism, having measured none").
- No rollout-level fidelity measure; named as the experiment that would settle generality.

### For the SO-101

This paper is cited in §7 of NEXT_ANGLES as the reason the PSG arm needs a matched-epochs control, and now you have its numbers: the control absorbed 100 % of the grounding gain. Three practices come out of it, all cheap. (1) **Report E_excess, not E_pred**, for W1/W2′: fit the decoder on encoder outputs of real frames, apply it to predictions; your G1 probes already fit the decoder, so this is one extra evaluation pass. (2) **Measure the planning noise floor on the untouched checkpoint across seeds** before any gate (your §20 — their number is 26 pp on the hardest task). (3) Never select a W1 checkpoint by loss for a control role; select by the gate it serves (G10 for the verifier, nothing yet for a planner). The honest sentence for an interview: "on LeWM, after the first epoch, one-step error and planning success are uncorrelated at r = −0.01 across fifteen checkpoints; I select world-model checkpoints by downstream gate, not by loss."

---

## 12c.4 Not All Errors Matter: Decision-Relevant Prediction Error (Wang, Fan, Huang; 2609.32322)

### The idea

Total prediction error averages over state dimensions; a planner reads only some of them. **DRPE** (eq. 2) is the same multi-step rollout error restricted (by a weight vector w, graded or binary) to the dimensions the decision depends on. The **iso-error protocol** holds total error fixed and moves it between relevant and irrelevant dimensions, so allocation — not magnitude — is the manipulated variable. Propositions: irrelevant error is provably free (Prop. 1); for any gap below the optimal-vs-blind-sequence value gap there exist two models with near-identical total error whose returns differ by more than it (Prop. 2); error below the planner's decision resolution is absorbed, which predicts flat-then-collapse (Prop. 3).

### Protocol

An 8×8 factored gridworld: 6 relevant dimensions (agent, key, carrying, door), 6 irrelevant (a static bit, a toggling bit, a random-walk distractor, a re-randomised distractor) — two of which are *perfectly predictable*, so a model can spend capacity on them. Exact V* by value iteration; graded relevance = expected |ΔV*| under one-dimension perturbation (0.29, 0.21, 0.15, 0.11, 0.15, 0.10; zeros on the irrelevant six). Planner: full-width depth-D search over model rollouts with V* on rounded relevant coordinates. Controlled models M(σ_R, σ_I) add Gaussian noise of scale σ_R / σ_I to relevant / irrelevant dimensions at every imagined step (noise compounds); 48 models (8×6 grid), **10 seeds × 30 episodes** each; plus 7 learned MLP variants; bootstrap CIs on correlations.

### Results

- Table 2: oracle 0.97; M(0, 0.3) E_tot 0.151, DRPE 0, success **0.97**; M(0.35, 0.1) E_tot 0.152, DRPE 0.234, success **0.37**. One per cent apart in total error, 60 points apart in success.
- Table 1: Spearman with success — total error −0.25 (all 55), −0.32 [−0.56, −0.04] (controlled 48); DRPE −0.84 / −0.98 [−1.00, −0.94]. A binary relevance mask does exactly as well as graded weights. Within the 77 iso-error pairs, ΔDRPE tracks Δsuccess at ρ = 0.97.
- Relevance is relational (§5.3): the same two noise models score 0.97 / 0.97 on task A and 0.82 / 0.97 on task B, where the static bit now selects the goal.
- Depth amplifies (Fig. 6a): irrelevant noise harmless at every planner depth; relevant noise worse the deeper the search.
- What averages miss (§5.4, Table 3): a model trained on 12k random transitions has the *worst* total error (0.355) and plans at 0.70; the rel-only model has nearly the *lowest* (0.102) and plans at 0.04, because its rollouts drift off its training manifold. The random-data model's failure is at one event (predicts carrying = 0.19 at pickup, below the 0.5 threshold) that contributes nothing to any average. Event-weighted training gives the best learned model (0.79).
- Negative result (App.): estimating relevance weights from a learned value function fails (39–42 % of mass on irrelevant dimensions; DRPE then correlates −0.24, same as total error). Mask sensitivity (Table 4): false negatives cost nothing (dropping 3 of 6 relevant dims keeps ρ = −0.98); **false positives are the risk** (adding one irrelevant bit drops ρ to −0.45).

### What is not there

- A 12-dimensional gridworld; "transfer to pixel-scale models is untested"; an exact V* heuristic.
- The learned-model pool (n = 7) is not ranked by either metric (+0.43 for both): the decoupling claim rests on the controlled family.

### For the SO-101

The abstract idea maps exactly onto G1b and G7: the state dimensions that decide bowl-in-bowl are the object's, and the commanded joints already determine the arm's, so a world model whose error sits on the arm's pose dimensions can have excellent E_tot and be useless for the decision. Three transfers. (1) **Report the per-dimension error profile of W1/W2′** on your probes (arm pose, gripper, object position when G7 labels exist), ordered by relevance, instead of one open-loop ratio. (2) The false-positive warning tells you how to build the G7 mask: exclude the arm-pose dimensions from the object-state probe even though they are "relevant-looking", since they carry prediction error the joints already explain. (3) The event-level failure (carrying flag at pickup) is your grasp onset: the one transition where W1's error decides everything and where an average over frames hides it — the same reason ContactGuard anchors at closure (12a.8).

---

## 12c.5 SALT: State-Affine Latent Transition (Zhang, Du, Zhen, Shao; 2609.33595)

### The decomposition (§3.2)

Rollout error δ_H = Σ_j Φ_{j→H}(ε_j + r_j) with Φ the product of state Jacobians of later transitions and r_j the non-linear remainder (Prop. 1; ‖r_t‖ ≤ (L/2)‖δ_{t−1}‖² under a Lipschitz Jacobian). For LeWM, the current-state Jacobian's top singular value varies strongly with the state (Fig. 1a) and the first-order reconstruction of the rollout error leaves 10 % unexplained at H = 5 and 49 % at H = 20 (Fig. 1b). Prop. 2: the Jacobian is state-independent iff the transition is affine in the state, F(z, c) = A(c)z + B(c); then r ≡ 0 and Φ_{j→H} = A(c_{H−1})⋯A(c_j), computable from the action sequence alone.

### Method

SALT: F(z, c) = [A_0 + Σ_r (W_g c)_r N_r] z + Bc + b, R = 16 dense modulation matrices, σ_max(A_0) < 1 via an orthogonal-times-tanh-diagonal parameterisation (eq. 19); 0.70M parameters against LeWM's 11.58M predictor; same ViT-Tiny encoder, projector, action encoder and SIGReg (λ = 0.09). **Recursive rollout training** (eq. 9): K = 5 predicted latents fed back and all supervised, six-frame windows at frameskip 5, targets not detached. Table 3 is the interaction that matters: rollout training lifts SALT 84.4 → 92.7 and *lowers* LeWM 82.7 → 77.8.

### Protocol and numbers

Four LeWM environments, CEM (300 candidates, 30 iterations, 30 elites), H = 5 with replanning every 5, goals 25 steps ahead; **one trained model per method, three evaluation seeds × 50 episodes, mean ± std**; reproduced LeWM as the matched control; literature rows quoted and flagged as non-comparable. Table 1: SALT 98.0 / 90.7 / 88.0 / **94.0** vs LeWM 90.7 / 85.3 / 87.3 / 67.3 (Two-Room / Reacher / PushT / Cube), avg 92.7 vs 82.7. Fig. 2a: SALT's one-step error is **1.48–2.19× LeWM's** in every environment. Fig. 2b–c: on PushT SALT's rollout error is *higher* for the first four steps and lower after; propagation-operator norm grows 4.28× over 20 steps vs 8.74× (App. C.3: 2.63 → 11.26 vs 2.46 → 21.51). Failure anatomy (§4.4, 150 paired episodes): "mismatch failures" (model-predicted cost rises >5× after execution and re-observation) 23.3 % → 2.0 %; unflagged 11.3 → 4.0; SALT succeeds on 43 episodes where LeWM fails, LeWM on none where SALT fails. Long-horizon (Table 7, H = 10–20, goals 100 steps ahead): SALT ahead in all twelve cells by 4–25 pp. Table 2: 8.1× faster per step, 3.1× faster CEM. Ablation (Table 8): removing the action modulation of A drops Reacher 90.7 → 11.3.

### What is not there

- Real robot: none. One training run per method (seeds are evaluation seeds).
- The σ_max(A_0) < 1 constraint is controlled (App. C.4: 4.28× → 4.42× without it) but global contractivity is explicitly not claimed.
- Which of the two ingredients (affine structure vs rollout training) does the work is answered only by Table 3's 2×2, with one run per cell.

### For the SO-101

Three things. (1) The "mismatch failure" statistic — the ratio of the replanning cost to the pre-execution cost — is a **free self-diagnostic** for any latent planner, computed from the model's own numbers, and it is the shape of your G1b/G6 failures: plans that look good in imagination and collapse on re-observation. Log it. (2) W1's predictor is LeWM's; rollout training *hurts* that architecture (Table 3), so the AR-term idea from 12a.4 should be tried with a small affine or near-affine transition, not bolted onto the transformer predictor. A 0.7M-parameter predictor also runs an order of magnitude faster on the Orin, which changes the monitor latency budget in 13.2. (3) The propagation-norm measurement (spectral norm of the Jacobian product along a demo action sequence) is one evening on W1's existing checkpoints and answers "does my model amplify errors" without a robot.

## 12c.5a When Does LeJEPA Learn a World Model? (Klindt, LeCun, Balestriero; 2605.26379; tested 2026-10-27, corrected 2026-10-28)

### The claim

An identifiability result for LeJEPA. Under stated premises on the world — independent, stationary, Gaussian latent factors following an Ornstein–Uhlenbeck process — a SIGReg-regularised JEPA recovers the true latent state up to a linear map. When the data is not like that, the prediction is specific: trajectory data that moves some factors much more than others gives **anisotropic** recovery — the factors the data varies most are recovered best (their Table 2, right). The paper's experiments are synthetic, where the true latent is known.

### Why this rig can test it

A real arm logs its true state. The six joint angles in `observation.state` are a ground-truth latent for the arm part of the scene, recorded at every frame, which no simulation-free paper usually has. So the consequence can be measured directly: regress the joints from W1's latent and look at the per-joint R².

### Protocol (bars `ad40675`, amendment `e6ba2fe`, both before any latent was computed)

Ridge regression h → joints, α by 5-fold cross-validation grouped by episode on the 157 training episodes, R² per joint on the 25 validation episodes. Bars: **B1** W1 mean R² < 0.80 (not full recovery, because teleop violates the premises); **B2** anisotropy (max − min per-joint R²) ≥ 0.20; **B3** the model with the higher R² has the higher G1b top-1 (direction only). The amendment came from an absent file: the W2 checkpoints of the 10-13 investigation were not on the Hub or the Orin, so the comparison became W1 vs W2′'s frozen DINOv2-S encoder, n = 2 for B3 — a sighting, not a test. Written before any number, and said so.

### Numbers (corrected 2026-10-28)

The first extract fed both encoders the red channel only (chapter 6.4a). Corrected, same scripts and bars: W1 per-joint R² pan 0.58, shoulder lift 0.91, elbow 0.88, wrist flex 0.27, wrist roll 0.30, gripper 0.45; **mean 0.565, anisotropy 0.639**. W2′'s DINOv2-S: 0.867 / 0.301. G1b top-1 W1 0.6046, W2′ 0.6458. B1, B2, B3 met; the verdict did not change, the numbers did, and both rows stay in the register.

### Reading

W1, trained on goal-directed teleop, recovers the joints the task moves most and loses the ones it barely varies: the anisotropic recovery the theorem predicts, on a real arm with the true state logged. The more important line for this book is B3: the higher-R² model is ranked higher on G1b, but both sit far below the action-only baseline (0.90), and a 0.30 R² gap moves top-1 by 0.04. **Identifiability is not what limits G1b here**; the decoy chunks differ in arm state, which the commanded joints already carry (chapter 12.4.1).

### What is not there

The premises are not tested; teleop violates them and that is the point of B1. ε and δ (the paper's distance-from-isotropy and drift statistics) are reported but only indicative, because the autocorrelation ρ̂ is estimated, not designed. Gate 1 — collect random-walk data on the arm and retrain, so that the premises hold and recovery should become isotropic — needs a robot day and vast; not run.

### For the SO-101

This is the cheapest real-robot test of a world-model theory available to anyone with a logged arm: one regression per joint. It also says what to do if W1 is ever retrained for planning: the wrist joints are nearly invisible to it, so a task that depends on wrist orientation needs data that varies the wrist.

## 12c.6 The three proxy papers, side by side

| | What varies | What is held fixed | Metric | n / seeds | Result |
|---|---|---|---|---|---|
| 2610.19375 | fine-tuning epochs; grounding vs continued | checkpoint family, planner, test set, planning seeds | E_excess, drift vs planning success | 3 train × 3 plan seeds, 50 traj | r = −0.01 after epoch 1; control absorbs grounding; noise floor 26 pp |
| 2609.32322 | error allocation at fixed total | planner, task, V* | DRPE vs total error vs success | 48 models × 10 seeds × 30 eps | 1 % total-error gap → 60-point success gap; ρ −0.98 vs −0.32 |
| 2609.33595 | transition structure × training objective | encoder, SIGReg, planner, data | one-step error, rollout error, Φ norm, success | 1 run × 3 eval seeds × 50 eps | worse one-step error, +10 pp success; mismatch failures 23 → 2 % |

The sentence the three make together, for chapter 3 and for an interview: *a world model's training loss measures something the planner does not read, under conditions the planner does not meet (teacher forcing, one step, on-distribution), so it cannot rank models for control; rank by the gate, measure the error where the decision is made, and always run the matched continued-training arm.*

## 12c.6a The bridge statistic: correlation of the proxy with the real outcome across models

All three papers above say that an offline number (one-step error, total prediction error) can fail to predict the decision that matters. The sim-to-real literature has a standard statistic for checking whether a proxy is usable at all, the **sim-to-real correlation coefficient** (SRCC; surveyed in Aljalbout et al., 2510.20808, §5.1): evaluate several policies or checkpoints both ways, and compute the Pearson correlation between the proxy scores and the real success rates. 2610.19375's r = −0.010 across 15 checkpoints is this statistic for one-step fidelity versus planning success.

On this rig, every offline evaluator should earn its place the same way before it is used to choose anything: W1 rollout error, a judge's score, a curation score, a world-model "simulator" (12d, route 5). Recipe: at least five checkpoints or policies with real closed-loop success measured on the same sealed layouts; the proxy computed on each; the correlation with a bootstrap interval over checkpoints; and the rank agreement (would the proxy have picked the same best checkpoint?). With five points the interval is wide, so state it; a proxy that cannot rank five policies is a measurement, not a selector.

## 12c.7 Corrections and additions

- 12.2: replace "SIGReg, a VICReg-like term" with the real description: a Cramér–Wold sketch of an Epps–Pulley characteristic-function test against N(0,1), bounded gradients, O(N), one hyper-parameter, derived as the risk-optimal embedding law (LeJEPA §3–4).
- 12a.3 / 12a.4: V-JEPA 2-AC's two-term loss is the origin of the AR terms; its released rollout loss is reported buggy by 2512.24497 — any reuse of the public checkpoint must be parity-checked against a *number*.
- 12.4.1 and §7: the matched continued-training control is now backed by the paper's own absorption result; the planning noise floor across seeds (26 pp) joins the §20 instrument check.
- G1/G7 reporting: per-dimension error profile and E_excess replace the single open-loop ratio; the G7 probe mask excludes arm-pose dimensions.
- Register: add SALT's mismatch-failure ratio as a logged quantity for any imagined-cost comparison.

- 12.2 / 12.4.1 (12c.5a): the joints are a logged ground-truth latent; W1 recovers them anisotropically (mean R² 0.565, wrist flex 0.27), as LeJEPA identifiability predicts for trajectory data, and the R² gap to DINOv2-S does not explain G1b. Report per-joint R² for any future world model on this data.

## 12c.8 Exercises

1. Write SIGReg from Algorithm 1 in 20 lines; run it on 1,000 W1 latents and on 1,000 random Gaussians; report both statistics and say what a 10× gap means.
2. Compute E_repr, E_pred, E_excess for W1 and W2′ on the G1 probe targets with the decoder fitted on real frames only. Report whether the protocol has power (E_excess / E_pred) before reporting the numbers.
3. Build the per-dimension error profile for W2′ on arm pose, gripper, and (once G7 labels exist) object position. Mark which dimensions the commanded joints already explain.
4. Measure the propagation-operator norm of W1 along 20 demo action sequences for H = 1…20; plot the growth factor; compare to SALT's 4.3× / LeWM's 8.7×.
5. From the V-JEPA 2 Table 2, write the Wilson intervals for every cell and say which differences between Octo and V-JEPA 2-AC are resolved at n = 20.
