# Chapter 13b. Which demonstrations help: the data papers, and the judge that chooses them

Chapter 13.3 says the limiting resource is labelled outcome and lists curation signals. This chapter is the reading behind a typical robot-data research posting ("which data improves robot learning, how to make it useful, when to introduce it"): four papers that answer it with experiments, plus one that shows what happens to the judge doing the choosing. Same format as 12a: method with the objective, protocol with n and seeds, numbers with table references, what was not run, and what it means for a SO-101 with 69 rollouts and a leader arm.

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | Quality-aware VLA pretraining (AXIS-Bench) | 2610.18869 | Weighting vs filtering vs conditioning × chunk vs episode × 3 VLAs; conditioning wins on average, rankings reverse by model |
| 2 | CUPID | 2506.19121 | Influence of each demo on *closed-loop return* via REINFORCE-weighted action influence; filters brittle strategies and spurious correlations that quality metrics promote |
| 3 | Quality over Quantity (QoQ) | 2603.09056 | Influence of each state–action pair on *validation loss*, max over validation pairs, trajectory-level selection; 99 % curation accuracy on Can-paired |
| 4 | Ego4WAM | 2609.40341 | Egocentric human data: alignment, duration vs diversity, video-only vs action supervision, usage stage; closed-loop evaluation throughout |
| 5 | Coverage shift in VLM monitoring | 2610.24801 | Better policy → fewer but harder failures; a fixed judge's conditional miss rate rises 31 → 50 % while absolute misses fall 5 → 0.9 % |

---

## 13b.1 What Matters in Quality-Aware VLA Pretraining (Yan et al., Georgia Tech / Axis Robotics et al.; 2610.18869)

### The question and the instrument

How should a *quality* signal enter pretraining so that the policy is better **after target-task fine-tuning**? The benchmark (AXIS-Bench) is 5,000 crowdsourced simulation teleop demonstrations with recorded object and contact states, so quality can be recomputed offline on the same trajectories; 24 target tasks; 354 FT demos; 50 fixed layouts per task → **1,200 trials per configuration**. Three VLAs (π0.5 from the DROID checkpoint; SmolVLA with frozen VLM and random-init expert; X-VLA), each compared only against itself with matched data, budget, FT set and layouts.

Two reward definitions (§IV-A): object-centric r_t = 0.8 p_t + 0.2 c_t (progress + end-effector proximity) and **approach-aware** R_t, which scores approach (normalised distance closed before first sustained contact) and manipulation (object progress after contact) separately with λ = 0.5, b = 0.1 (eq. 3). Labels: chunk-level Δ_t = reward change over H = 10 frames, truncated at the contact boundary; episode-level P_e = fraction of retained steps with Δ_t > 0 (eq. 5); quintile bins for tags. Human check: three annotators mark unproductive intervals on 50 demos; Spearman ρ(P_e, human) = 0.612 (object-centric) / 0.526 (approach-aware).

Three ways in (§IV-C): **weighting** w_t = min(exp(10Δ_t), 2) (AWR-style); **filtering** top-50 % by chunk or episode, repeated to match updates; **conditioning** by a `Quality: q` tag appended to the instruction (π0.7-style), tag dropout 0.15 / 0.05, FT with constant tag 5, inference with tag 5.

### Results

- Plain-BC references: π0.5 57.42 %, SmolVLA 5.08 %, X-VLA 25.75 %; FT-only (no AXIS pretraining) 8.83 / 2.50 / 6.17.
- Finding 1 (Fig. 5): approach-aware **episode conditioning** beats plain BC on all three: +1.0 %, **+73.8 %**, +8.1 % relative (SmolVLA 5.08 → 8.83 absolute). But rankings differ: π0.5 conditioning ≈ weighting > filtering; SmolVLA favours conditioning; X-VLA favours filtering (all four matched settings) and weighting.
- Finding 2: conditioning favours episode labels (5 of 6 settings), filtering favours chunk labels (4 of 6), with reversals under the object-centric reward.
- Finding 3 (Fig. 7): after conditioned pretraining, **tagged FT** beats plain FT under both inference modes (tag-5 inference: +0.25 / +5.00 / +2.17 pp); adding tag 5 only at inference can hurt (SmolVLA −2.42 after plain FT).
- Filtering vs random at the same half-data budget (Fig. 4c): approach-aware episode filtering +1.08 / +5.75 pp over random (SmolVLA / X-VLA); object-centric chunk filtering +3.17 for SmolVLA but **−3.00 for X-VLA**. Quality filtering is not reliably better than random.
- Three-seed replication (Table I): conditioning − BC = +3.86 ± 5.83 (π0.5), +23.42 ± 45.42 (SmolVLA), +5.62 ± 5.39 (X-VLA) — positive on average, with SDs larger than means for two of three.
- Real world (Fig. 9, Franka FR3, 10 tasks × 24 trials, π0.5-base): FT-only 9.6 % → simulation-pretrained BC 31.7 % → conditioned 32.5 %. The +22 pp is *pretraining at all*; quality conditioning adds 0.8 pp (5 tasks up, 4 down, 1 tied).

### What is not there

- Seeds on the main sweep (one training seed; Table I's three seeds show SDs of the same size as the effects).
- A matched control for conditioning's data coverage (conditioning keeps all data; filtering discards half) — stated as future work.
- The real-world gain from *quality* is 0.8 pp on 240 trials; the paper says so.

### For the SO-101

This is prep-queue 3.2 done for you at a scale you cannot afford, and its answer is sobering in the right way: the big number is pretraining (9.6 → 31.7 %), the quality mechanism adds ~1 pp on hardware and reverses sign across models. Two things transfer. (1) The **approach-aware reward** (eq. 3) is computable on your episodes from the key-press labels and the G9c-3 end state (first gripper close = contact boundary; nesting = progress), and P_e is a better per-episode quality number than length. (2) SmolVLA with a frozen VLM is the model that gained most from conditioning (+74 % relative, from a 5 % base) — the same frozen-encoder bottleneck SEVO found (14a.7). If you ever tag your 69 + tidy-loop episodes, condition, don't filter: at your n, filtering halves the data and the paper shows it loses to random as often as it wins.

---

## 13b.2 CUPID (Agia, Sinha, Yang, Antonova, Pavone, Nishimura, Itkina, Bohg; Stanford / TRI; 2506.19121)

### The idea

Define curation as maximising closed-loop return, not a quality heuristic. The **performance influence** of a demonstration ξ (Def. 1) is dJ(π_θ)/dε for the BC objective perturbed by ε·ℓ_traj(ξ). It cannot be computed directly (J needs the dynamics), but Prop. 1 shows it equals an expectation over the policy's own rollouts τ of R(τ)/H × Σ_{(s',a')∈τ} Σ_{(s,a)∈ξ} Ψ_a-inf((s',a'),(s,a)) — the **action influence** of each demo transition on each rollout action's log-likelihood (Def. 2, eq. 2), weighted by the rollout's return (R = ±1). So: collect m rollouts with outcomes, compute influence scores with TRAK (random projections + Gauss–Newton Hessian), average. Filter-k = drop the least influential; select-k = add the most influential from a holdout pool. For diffusion policies the log-likelihood gradient is replaced by the gradient of a label-agnostic squared-output surrogate (App. A.1, following Zheng et al.), which they find works where the naive denoising-loss substitution does not. CUPID-Quality adds a reward-agnostic outlier score Ψ_qual (eq. 6) at α = ½.

### Protocol

RoboMimic MH (Lift, Square, Transport; 300 demos with ground-truth quality tiers), diffusion policy; filter-k from a random 2/3 subset, select-k from a random 1/3; **m = 100 rollouts** for curation; success over **500 rollouts averaged over the last 10 checkpoints, 3 seeds (500 policies trained in total)**. Real Franka FR3 (image policy, 25 rollouts for curation, 25 for evaluation with the last checkpoint): Figure-8 (160 demos in 4 quality tiers, a rope cleat hitch), TuckBox (120 demos, 2:1 sliding : pick-and-place, test-time mass shift makes sliding fail), Bookshelf (120 demos with a background colour spuriously correlated with the pulling strategy). Baselines: DemInf (mutual information, offline), Demo-SCORE (classifier on success vs failure states), Success Similarity, Random, Oracle (ground-truth labels).

### Results

- RoboMimic (Fig. 3): DemInf curates the highest *quality* by the labels and does not win on *success*; CUPID matches or beats it; CUPID-Quality beats the Oracle in 3/5 and exceeds the official diffusion-policy number on Transport with < 33 % of the demos. Lift is redundant (15 % of data suffices); on Square removing almost anything hurts.
- Figure-8 (Fig. 4a, filter 66 %): base 36 → CUPID 72 → CUPID-Quality 80 (Oracle 84; DemInf 64; Demo-SCORE 56).
- **TuckBox** (Fig. 4b): base policy 0 % under the shift (slides every time); Demo-SCORE inapplicable (no successes); DemInf 0 %, CUPID-Quality 4 % — both keep the "cleaner" sliding demos; **CUPID 84 %** (Oracle 88) by promoting pick-and-place from the failures alone.
- **Bookshelf** (Fig. 4c): base 44; Demo-SCORE 44 (misattributes failure to the stacked book, the rollout correlate, not the white background, the cause); DemInf 36; CUPID-Quality 20; **CUPID 84** (Oracle 96).
- Rollout budget (Fig. 6): rankings stabilise at m ∈ [25, 50] on Lift/Square, [50, 100] on Transport.
- Transfer (Fig. 7 / 15): datasets curated with a diffusion policy improve π0 LoRA post-training: Figure-8 48 → 92 (filter), TuckBox 36 → 64. "Scaling pretraining does not make a VLA ignore brittle strategies in its fine-tuning data."

### What is not there

- Intervals on the 25-rollout real-robot numbers (84 vs 88 is 21 vs 22 successes).
- k is chosen by the experimenter; how much to curate is an open problem (§9).
- Cost ≈ one policy training per curation pass; group effects ignored (greedy top-k).

### For the SO-101

The two hardware results are the ones that matter for you, and both are about *failures*. TuckBox: a quality metric prefers the smoother strategy, which is the brittle one; only outcome-attribution finds the robust one, and it does so from failures alone. Bookshelf: a success-vs-failure classifier on rollout states (the Demo-SCORE / Success-Similarity family, which is also what any "judge on rollouts" curation would do) blames the correlate, not the cause. Your bowl data has exactly this structure — two bowls, one mat, a side decision visible in 6 % of frames — and fact 7 is a spurious correlation between pose and target. CUPID's requirement is m ≈ 25–50 labelled rollouts of the policy you want to improve; you have 69. The experiment is: fit the influence scores for the 69 rollouts against the training episodes, filter 33 %, retrain, paired evaluation on the robot day. The baselines to run beside it are the cheap ones the paper beat: length-based filtering and random at the same budget.

---

## 13b.3 Quality over Quantity (Lee, Min, Kim, Kang, Liu, Pinto, Lee; KAIST / Berkeley / NYU; 2603.09056)

### The idea

Same tool, different target: influence on **validation loss** over a small set of desirable demonstrations (10–20), not on closed-loop return. Two changes to make influence functions work on robot data (§IV-B): (1) **maximum** gradient product over validation pairs instead of the sum, because a validation set holds many behaviours and most are irrelevant to a given training pair (eq. 6: QoQ(s,a) = max_{(s',a')∈D_val} g(s',a')ᵀ g(s,a), with g the ℓ2-normalised log-likelihood gradient, first-order/TracIn-style, no Hessian); (2) **trajectory-level** aggregation (mean over the trajectory, select top-N), because state–action-level selection picks redundant grasp frames and loses reaching and multimodal strategies. Gradients from a subset of layers (the action head suffices: 83.6 vs 82.1 all layers, Table II), compressed with OPORP. For flow-matching VLAs (GR00T N1), the flow loss gradient stands in for the log-likelihood.

### Protocol and numbers

- Sim (RoboMimic Can-paired: 100 successes + 100 failures, 10 successful validation demos): **curation accuracy** (fraction of successful trajectories in the curated set) 99.4 ± 0.3 vs Behavior Retrieval 67.8, Flow Retrieval 56.9; policy success 99.2 % vs best baseline 76.0 (5 runs).
- Real Franka, GR00T N1 LoRA, 3 runs: banana grasping (60 S / 40 F, 10 validation) curation accuracy 83.6 vs 67.3; success 86.7 vs 56.7 (all data 36.7). Multi-object (80 demos of four other objects; validation = 10 banana demos): top-50 % → 93.3 % vs all data 56.7, Behavior Retrieval 20.0.
- DROID in the wild (200 pen/pencil trajectories, 133 S / 67 F): curation accuracy 78.2 vs 66.4 / 60.4 / 66.5.
- **Rollouts as validation** (§V-D): 20 rollouts of the uncurated policy (5 S / 15 F); score = QoQ against successes minus QoQ against failures, weighted by set size; Open-cabinet 25.0 → 43.3 %.
- Ablations (Fig. 6): max vs sum scoring 83.6 vs 58.3 accuracy; trajectory vs state–action curation 86.7 vs 26.7 success. Budget (Table III): 86.7 at N = 60 (the true success count), 36.7 at N = 10. Kendall's W across seeds 0.77 vs 0.33 (Behavior Retrieval).

### What is not there

- Real-robot n: 3 runs; the SDs in Table III are ±10–27 points.
- Comparison with CUPID is argued (§II-B: CUPID with success-only validation reduces to the *sum*, which they show is unstable) but not run head-to-head.
- Shared embodiment between train and validation is assumed.

### For the SO-101

The cheapest of the three curation methods to run: no rollouts needed in the base form, 10 good demonstrations as the validation set, gradients from SmolVLA's action expert only, one pass. And the §V-D variant turns your 69 labelled rollouts into the validation set directly (successes positive, failures negative). The two design rules are the ones to keep whatever you run: **max over validation pairs** (so the 94 % of frames where the instruction is redundant do not swamp the 6 % where it is not) and **select whole episodes** (so reaching and the two strategies survive). The number to report first is the paper's: curation accuracy against your own labels, before any policy is retrained.

---

## 13b.4 Ego4WAM (Sun, Liu, Wang et al.; Fudan / Horizon Robotics; 2609.40341)

### The question

Egocentric human data "scales", but which properties drive robot gains: **alignment** (viewpoint, speed, style matched to the robot), **duration vs task diversity**, **supervision** (reliable hand-pose actions vs video only), and **usage stage** (pre-, mid-, post-training)? Held fixed: a world-action model backbone (FastWAM-style: a video branch for future-frame prediction and an action branch, shared attention, flow matching on both; proprio only to the action branch), a 32-D unified bimanual action space (per arm: 6 joints, 1 gripper, 9-D EEF), and human actions supervising only the shared EEF + gripper dimensions with inactive dimensions kept on an analytic noise path (App. B.1).

The **egocentric data pyramid** (Fig. 1): ~4 h robot-aligned video-action (EgoAlign, their own, head-mounted camera, hand pose via VGGT-Omega + GroundingDINO + HaMeR, filtered); ~12k h multi-task video-action after spike/projection filters (EgoDex, EgoLive, EgoSuite …); ~120k h video-only (Egocentric-10K/100K, RekaDaily). Three stages: video pre-training (world loss only, 15k h), mid-training (video + action on the 12k h plus ~90 h of robot data from two unrelated task families), post-training on the target robot task.

### Protocol and numbers

- Real bimanual Piper, **Place into the Basket** and **Fold Cloth**: 300 ID robot demos; a matched human set of 300 ID + 100 object-OOD + 100 scene-OOD. Success and a normalised progress score (milestones; 20 trials per cell read from Fig. 3–4).
- Alignment (Fig. 3, Basket): robot-only 60 / 10 / 0 % (ID / object-OOD / scene-OOD); + EgoAlign ID 60 / 30 / 10; + object-OOD human demos **object-OOD 10 → 60**; + scene-OOD human demos scene-OOD 0 → 20 (progress 0 → 47.5).
- Robot-data budget (Fig. 4, Fold Cloth): 300 robot demos 100 / 20 / 60; 100 demos 40 / 0 / 0; **100 demos + 500 human 80 / 60 / 80**; 20 demos + human 0 / 0 / 0; **multi-task mid-training + 20 demos + human 80 / 80 / 60**. Human data reduces the robot-data requirement 3× but cannot replace it, and the mid-trained prior is what makes 20 demos usable.
- Duration vs diversity (Fig. 5, RoboDojo, same steps): fixing 500 tasks and going 1k → 4k hours lifts Generalisation 5.0 → 7.8 and Long-horizon 12.0 → 16.2; going 500 → 6k tasks at matched hours-per-task lifts Generalisation to 7.0 and Long-horizon to 16.2 but **lowers Precision**; at matched volume, denser coverage of common tasks beats broader coverage on several aggregates. Per-task transfer is both positive and negative (Fig. 7).
- Supervision (Table 1, RoboDojo avg score / SR): none 6.39 / 3.15; mid-train only 10.28 / 6.42; **video-only pre-train 14.13 / 9.45**; pre + mid 15.04 / 10.78; pre-train + joint attention 20.29 / 14.35. Video without actions is worth more than action-labelled mid-training alone, and the two overlap.
- Table 2: Ego4WAM at 0.1k h robot + 15k h ego scores 20.29 / 14.35, beside Xiaomi-1 at 7.2k h robot + 100k h ego (20.07 / 13.93) and GalaxeaVLA-0.5 (20.23 / 14.88).

### What is not there

- Intervals or seeds on the real-robot cells (20 trials each, read from bar charts); RoboDojo numbers are single runs.
- Compute matching between video-only pre-training (15k h) and mid-training (12k h) — stated as deliberately not matched.
- Any monocular-vs-stereo or hand-pose-error analysis of the alignment pipeline.

### For the SO-101

This is the vocabulary for interview question 16.6 ("egocentric / UMI / retargeting?"), with numbers. The honest answers it gives you: (1) alignment matters more than hours — 100 object-OOD human demos moved object-OOD success from 10 to 60 % when 300 robot demos could not; (2) human data buys a 3× reduction in robot demos, not a replacement; the floor was 20 robot demos *with* a multi-task prior; (3) video-only pre-training on a world-model branch is worth more than action labels on a smaller set, which is the strongest external argument for the world-model-first design your stack already has; (4) diversity can hurt precision. For your rig the practical move is small and matches (1): a phone or head camera recording *you* doing the bowl task on the same mat is "robot-aligned egocentric data" in this paper's sense, and it needs no hand-pose pipeline if it only feeds the world model (video-only supervision, Table 1). That is a one-session experiment with a W1 retrain and the G10 probe as the gate.

---

## 13b.5 Coverage Shift in VLM Success Monitoring (Guan, Zhang, Lyu, Chen; HKUST / NTU / PolyU; 2610.24801)

### The claim

A fixed VLM judge, evaluated on the failures a policy produces, can get *worse at detecting the failures that remain* as the policy improves, even though it makes fewer wrong accepts per execution. Identity (eq. 1): accepted failures per execution m_t = b_t · c_t, failure incidence × conditional acceptance; learning lowers b_t and can raise c_t, because it changes *which* failures reach the judge (eq. 2).

### Protocol

ManiSkill3 Franka, stack and tray tasks; a scripted controller whose parameters are sampled from a fitted distribution (branch, offsets, release height, release, retreat); each round executes 100 samples, keeps some by a filter, refits. The **central comparison uses simulator truth for selection** (arm C1), so the judge only *watches*: N0 (no learning), R0 (refit on everything), C1 (truth-filtered); **20 seeds, rounds 1–5 pooled, paired two-sided 95 % t intervals**. Judges: Qwen3-VL-8B (J1), Qwen3.8-27B (M1), RoboReward-8B, a DINOv2 logistic head, and five closed models, on **348 executed end states × 4 views = 1,392 stimuli** with ordered simulator predicates (hover > 6 mm, held_on, behind, edge, beside). A preregistered boundary probe F1: the same 40 scenes with the grasped cube above vs below the tray rim.

### Results

- Table 2 (M1 / tray / front-low view): success 66.8 → 97.0 %; hover incidence 15.9 → 1.8 %; accepted hovers per execution **5.0 → 0.9 %**; **acceptance given a hover 31.1 → 50.0 %** (paired +18.9 pp [9.7, 28.1]); below-rim share of residual hovers 31.3 → 52.6 % (+21.4 [11.7, 31.1]). Refitting alone (R0) contributes +10.3 [5.2, 15.4] of the conditional rise; truth selection adds +8.6 [1.4, 15.8].
- F1 probe: the judge accepts **0/160 above-rim hovers and 148/160 below-rim hovers**. The static diagnostic set contained only above-rim hovers, so the error map predicted "no blind spot" for a cell where 81 % of accepted loop failures were hovers.
- Error maps (Fig. 3) are view- and type-dependent: hover acceptance concentrates in the oblique view, behind-acceptance in the front-low view; M1 has lower aggregate false acceptance than J1 (9.8 vs 21.1 %) but a different composition.
- When the judge *does* select data (C3) the effects are heterogeneous (Table 3): J1/tray/front-low leaves 15.4 % "behind" failures vs 1.2 % under a within-stratum label permutation; a second checking view recovers 94.0 % where a stricter threshold does not (Table 4); a threshold at 0.995 that keeps a third of true successes *lowers* final success 91.4 → 72.8.
- The preregistered static-prediction (T1) and repair-criterion (T3) tests **fail**, and the paper says so.

### What is not there

- Two simulated tasks, a scripted low-dimensional policy family; one judge–task–view cell for the central result; a single-seed diffusion-policy pilot not expanded.
- The mechanism is localised (shared height parameters, release cap) but not isolated.

### For the SO-101

This is the paper behind the "coverage shift" line in 13.1.1, and it changes the judge rules in three concrete ways. (1) **Report four numbers, not one**: success, failure incidence, accepted failures per execution, and acceptance conditional on each failure type, with counts — your verifier card (NEXT_ANGLES §22) gets the last two columns. (2) **Keep boundary probes**: the G9c-3 masks and Flora were tuned on the failures of one policy generation (wrong object, dropped bowl, no grasp); the failures of a better policy will be *near* success (bowl on the rim, nested but tilted), which is the below-rim case. Before the flywheel trusts a judge, stage those boundary cases by hand — 40 scenes like F1 — and measure acceptance on them. (3) **A second view beats a stricter threshold** (Table 4): for your rig, that is the wrist camera as a *checking view at rest* for the top-camera judge, which costs one more G9c-3 call and nothing else. The field-level sentence: a judge's error rate is a property of the distribution it is measured on, and that distribution moves every time the policy is retrained.

---

## 13b.6 How this family evaluates, side by side

| Paper | Platform | Real n | Seeds / intervals | Matched control | Cheap baseline |
|---|---|---|---|---|---|
| AXIS-Bench | sim 24 tasks; Franka 10 tasks | 24 trials/task (240) | 1 seed main; 3-seed table with SDs | plain BC same data/budget | **random at same budget** (Fig. 4c) |
| CUPID | RoboMimic; Franka ×3 | 25 rollouts/task | 3 seeds sim (500 rollouts each); none real | Oracle with ground-truth labels | Random; quality metrics |
| QoQ | RoboMimic; Franka ×3; DROID | 3 runs/task | 5 runs sim | All data | retrieval baselines |
| Ego4WAM | Piper ×2 tasks; RoboDojo | 20 trials/cell | none | robot-only, same steps | none |
| Coverage shift | ManiSkill | — | **20 seeds, paired 95 % t; preregistered** | N0 / R0 / C1 truth-filtered | static error map (fails T1) |

Read with chapters 3–7:

- The curation papers agree on one thing against intuition: **a quality metric is not an outcome metric**. DemInf curates the "best" data and loses on success (CUPID Fig. 3); quality filtering loses to random on one VLA (AXIS Fig. 4c); quality-weighting promotes the brittle strategy (TuckBox). Your rule in 13.3.2 — test any curation signal as a judge first, then as a training weight — is what these three papers did and what their baselines did not.
- Random-at-the-same-budget is the baseline that keeps appearing and keeps winning something. It is free. It goes beside length in every curation bar.
- The real-robot gains attributed to *quality* are small on hardware (AXIS +0.8 pp on 240 trials) or come from failures and spurious correlations (CUPID), not from "better demos". On your data the second kind is the one you have.
- Coverage shift is the only paper in the whole knowledge base with preregistration, 20 seeds and paired intervals on its central claim, and it reports two failed preregistered tests. It is the standard for any judge claim you make.

## 13b.7 Corrections and additions

- 13.3.2: replace the curation-signal list with the three estimators — approach-aware progress P_e (computable from your labels), influence on validation loss (QoQ, no rollouts), influence on closed-loop return (CUPID, 25–50 labelled rollouts) — and the rule that random-at-budget is a baseline.
- 13.1.1 tiers: judge cards report accepted-failures-per-execution *and* acceptance-conditional-on-failure-type with counts; boundary probes staged by hand before the flywheel; the wrist camera as a checking view at rest.
- 16.6 ("egocentric / UMI?"): replace "honest: no hands-on" with Ego4WAM's four numbers and the one-session proposal (your own egocentric bowl video into W1, video-only supervision, G10 as gate).
- NEXT_ANGLES: new candidate — **QoQ on the 69 rollouts** (validation = 10 best successes; report curation accuracy against labels first; then CUPID with the rollouts as the return signal; retrain; paired robot-day evaluation against random-at-budget and length-filtering).
- Prep queue 3.2: superseded by AXIS-Bench; keep only the conditioning arm (SmolVLA + episode tag), since filtering at your n is the arm the paper shows losing to random.

## 13b.8 Exercises

1. Compute the approach-aware reward (eq. 3 of AXIS) for 20 of your episodes from the key-press and G9c-3 signals; compute P_e; rank the episodes; compare the ranking to length and to your own judgment of five of them.
2. Implement QoQ scoring on SmolVLA's action-expert gradients for the 69 rollouts with the 10 cleanest successes as validation; report curation accuracy against the labels and Kendall's W over three gradient subsets.
3. Write the CUPID protocol for the bowl data as a NEXT_ANGLES entry: m, k, the random and length baselines, the paired robot-day evaluation, the bar.
4. Stage 40 boundary scenes for G9c-3 (bowl on rim, nested but tilted, nested with the wrong bowl) and measure acceptance; write the four-number judge report.
5. From Ego4WAM Fig. 4, write the sentence you would say to "would human video help on a $300 arm?", with the numbers and the one experiment you would run first.
