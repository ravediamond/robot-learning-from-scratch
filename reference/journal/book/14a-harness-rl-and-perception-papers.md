# Chapter 14a. Harness, deployment-time learning, and perception papers, in depth

Chapter 14 describes the harness as a lab instrument and the training vocabulary you need. This chapter is the reading behind it: thirteen papers in three groups. The first group (RegenHarness, HexaShell, Spotter is in 13a) are harness designs; the second (SEAR, Robo-COP, Mulligan, ReTrace, COAP/ETM, HyMeS, residual-foresight RL, HIR-ResRL) are ways a robot improves during deployment, each with a protocol for proving that it did; the third (SEVO, PEEK, blind assembly) are about what the policy sees. Each paper: method, protocol with n, numbers with table references, what was not run, and what it means for a SO-101 on an Orin with two webcams.

The one idea that runs through the deployment-learning group is SEAR's Table 2: five ways a self-improving system can look better than it is (more attempts, familiar tasks, privileged information, luck, larger budget), and the control for each. Hold every "the robot got better" claim, including your own flywheel, against those five.

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | RegenHarness | 2609.27612 | Evidence-gated harness: controller done ≠ node success ≠ goal success; role-isolated contexts; RSI with regression set |
| 2 | HexaShell | 2610.23479 | Six-layer agent harness; 739 s agent for 26 s robot; the latency number behind your design |
| 3 | SEAR | 2610.18792 | Benchmark for self-evolving agents; best-of-3, sealed judge, five tiers; practice ≠ test on hardware |
| 4 | Robo-COP | 2610.09228 | Orchestrator and policy co-evolve; skill-level curation from failed episodes; verify before adopt |
| 5 | Mulligan | 2610.05882 | Choose next round's initial states from observed failures + coverage fill; +10–34 pp real |
| 6 | ReTrace | 2610.15814 | Rewind to pre-failure state and re-execute; recovery data from the robot's own failures |
| 7 | COAP / ETM | 2610.12369 | Experience-to-memory: operator corrections become retrievable context for a frozen VLA |
| 8 | HyMeS | 2608.09410 | Memory as executable code steering a Markovian VLA; **SO-101** 25.7 → 57.1 % TSR |
| 9 | Residual-foresight RL | 2607.16506 | Residual policy over a frozen VLA with a world-model foresight critic; long horizon |
| 10 | HIR-ResRL | 2606.22860 | Hierarchical residual RL with human interventions as reward |
| 11 | SEVO | 2605.11114 | Observation-space fixes on SO-101: body-fixed cams, red LED, YOLO overlay; wrist cams hurt |
| 12 | PEEK | 2509.18282 | VLM draws path + mask on the image; policy learns only "how" |
| 13 | Blind precise assembly | 2610.23849 | Proprioception-only peg insertion, sim-to-real 73 % of 237 trials, Wilson bars |

---

## 14a.1 RegenHarness (Wang, Jie, Yan, Heng, Li; Country Garden / HUST; 2609.27612)

### What it is

A runtime contract, not a model. Task g compiles into T = (g, G, C_g, S, R): a dependency graph of subgoals with typed target effects, original completion conditions, safety/resource constraints, recovery budgets (eq. 2). Two fact sets per state version k: **observed** O_k (fresh, provenance known, meaning unverified) and **committed** C_k (admitted by the verification path). A skill invocation produces an evidence packet Z_k = (I, k_ctx, O_before, O_after, P, Δ̂_k) (eq. 5) and a *proposed* fact update; only a **satisfied** verdict with matching task / node / invocation / state-version identities may advance C. The verdict vocabulary is five words: satisfied, violated, ambiguous, stale, unsafe. Four role-isolated contexts compiled from one versioned memory (Table 1): planner C1, skill C2, verifier C3, recovery C4 — and C3 never inherits the planner's completion claim ("the cup has been grasped" is an intention in C2, not evidence in C3). Eq. 6 is the sentence to remember: **controller done ⇏ node success ⇏ goal success.**

Mechanics: idempotent dispatch keyed by task/node/attempt; resource leases with owner and expiry; deadlines; budgeted recovery with an escalation ladder per failure class (target_not_found → re-perceive → adjust view; no_progress → local replan → switch skill; unsafe_action → safe stop → human). Event-sourced memory with `supersedes` relations. Routing score s(π_i) = w_p p̂ − w_r r̂ − w_u û − w_l l̂ − w_c ĉ − w_h ĥ (eq. 4) with ĥ a penalty from transition residuals (expected vs observed effect). A world model is "optional support": predicted effects are decision inputs and "cannot establish post-execution success".

RSI (§4.9): configuration H_j (context rules, templates, routing, recovery policies) revised from mission records into H̃_j; released only if an acceptance predicate A over a **fixed regression set** R_j passes (eq. 7): safety and evidence contracts preserved, no regression on designated checks, improvement on a pre-declared metric, authorisation. The updater cannot weaken its own gate. Mission-pinned configurations; rollback to the last accepted version.

### Protocol

A quadruped deployment (voice-triggered warehouse navigation, inspection, analysis, message delivery, return, spoken report) with linked audio, images, trajectories and receipts; and "a separate circuit" showing completion depends on execution history, not endpoint proximity. No success rates, no n, no baseline harness. It is a design paper with one documented deployment.

### For the SO-101

Your v1.5 spec already has idempotent dispatch, two-field results and budgets (chapter 14.2.1); RegenHarness adds four things worth copying verbatim. (1) The **five-word verdict vocabulary** with *stale* and *ambiguous* as first-class outcomes: your G9c-3 "invalid" is *ambiguous*, and "the judge ran on a frame from before the last action" is *stale* — name both in the record. (2) **Role-isolated contexts**: the verifier prompt must not contain the agent's plan; write that as a rule in the harness and test it (feed the verifier a wrong plan; the verdict must not move). (3) The **regression set for configuration changes**: every edit to a verifier threshold, a budget or a prompt is a candidate H̃ that must pass a frozen set of recorded trials before it is live — your register's "frozen gate" idea applied to the harness itself. (4) The **transition residual** as a routing input: expected effect vs observed effect per executor, which is the per-executor report card you planned for G24.

---

## 14a.2 HexaShell (2610.23479)

Read in the earlier pass for one number, which is the reason chapter 14.2 is built the way it is: 739 s of agent time per 26 s of robot motion when a language-model agent sits in the loop. The design that follows from it — the agent dispatches unit actions and reads results at rest, never in the motion loop — is the same conclusion Spotter reaches with success numbers (13a.9, −2.7 / −3.8 for VLM-led) and SEAR reaches with time (14a.3: 12.8 min per live episode vs 3.8 for a frozen code policy). Three independent measurements of the same fact; cite all three when asked "why isn't the VLM in the loop?"

---

## 14a.3 SEAR (Wang, Li, Wang, Zhang, Cui, Tang, Buch, Xu; Luma AI; 2610.18792)

### What it is

A framework and benchmark for agents that improve robots from their own experience, in three modes: **code-as-policy** (the LM writes a program; no LM at run time), **agent-as-policy** (the LM chooses every command live), **auto-research** (the LM fine-tunes π0.5 on rollouts it collects from its code policies). Every agent starts from the same toolbox (SAM3, pretrained π0.5, IK, servo interfaces); what persists across tasks is a **library** of versioned skills and written memories. 330 simulated tasks in five families (robosuite 7, LIBERO 40, LIBERO-PRO 160, LIBERO-RIG 18 with hidden hardware faults, RoboLab 105 in Isaac Sim) and five physical tasks on two YAM arms. Seven LMs; over four years of agent runtime.

### The protocol (§2.2–2.4), which is the reason the paper is in the book

Table 2, five ways a self-evolving agent can look better than it is, each paired with a control:

| Looks like learning | Control |
|---|---|
| More attempts | one long session vs **best-of-3** restarts at a third of the budget; with vs without library (Lineage vs Vanilla) |
| Familiar tasks | scores on TEST initial states never shown to the agent; five tiers of distance from practice |
| Privileged information | a **sealed judge** grades the frozen policy with state and logic the agent cannot reach |
| Luck | **three independent repetitions** of the whole run |
| A larger budget | development time matched in every comparison; tokens, env steps and GPU time logged |

Sealed judge: the agent sees DEV seeds freely, VAL seeds are graded by the judge during development, TEST seeds once at the end; reported scores are TEST success over ten rollouts per task. "Agents still found ways around the seal, such as reading object geometry from asset files; §A.9 lists them." Generalisation tiers T1–T5: new initial states; same tasks under changed scenes or faults (frozen, no practice); new tasks in the same simulator after self-exploration; new simulator and action API; the physical robot. Physical protocol: nothing resets itself, so each task comes with a reset task the agent must also learn; practice stops after ten consecutive successes or 12 h; the frozen policy runs once on ten operator-staged scenes.

### Numbers

- Fig. 4–5: one long session beats best-of-3 when the model's time to first success exceeds a short attempt's length (Opus +31.9, DeepSeek +20, Kimi +14.5; GPT-5.6-sol +2.9 because it succeeds in 10 min). "Task difficulty alone does not decide which setup is better."
- Fig. 6: the library helps most in the middle — +0.9 above 80 % success without it, +12.0 between 30 and 80 %, +3.0 below 30 %. GPT-5.6-sol on RoboLab: 46.9 no library, 42.4 own library, **60.6 with Opus's library** — content matters more than ownership.
- §3.3, what carries forward includes mistakes: one Opus run built a placement verifier calibrated on 16 episodes whose three failures happened not to move; it declared success on 699 later rollouts, **wrong on 91 %**, and because it was also the stopping rule, 637 rollouts ended early; the run scored 4.6 %, siblings 15.5 and 23.6. Two DeepSeek runs with identical setup differed by 20.9 points. "A single run cannot be told apart from luck."
- Table 3, Opus code-as-policy by tier: T1 87.8, T2 59.7, T3 33.3, T4 28.7, T5 35.6; VLA auto-research 73.2 / 55.3 / 43.1 / 10.2.
- Fig. 9b: under a moved scene camera, code policy 9.0 vs VLA 68.9; under the seven faults that leave the camera in place, 90.4 vs 78.6. "Weights survive a moved camera; code does not."
- Table 4, physical (ten staged scenes per task, one practice run per agent): code-as-policy Fable 5.1 70.2, GPT-6-astra 41.4, Opus 5 35.6; agent-as-policy Fable 66.4, GPT-6-astra 88.0 (live episode median 12.8 min vs 3.8 min frozen). Fig. 11c: hard starts (Push-T needing > 90° turn) are 10/64 in practice vs 6/10 in test — **the agent's own resets avoid the hard states**; ten practice successes in a row ended practice on six runs and only one of them then passed all ten staged scenes. §4.4: in 29 of 34 failed Fable evaluations the policy stopped at a limit the agent had set itself after one incident ("a lesson learned once becomes a rule, and nothing in the loop tests it again").

### What is not there

- Intervals on the physical table (ten scenes, one run); the simulation spread is given as three-run ranges.
- A fixed-policy baseline on the physical tasks (what does π0.5 alone score on the ten staged scenes?).

### For the SO-101

This is the paper to read before any flywheel claim, and three of its findings apply to your repo today. (1) **"A verifier calibrated on a handful of episodes, used as the stopping rule"** is the exact failure your tier rules exist to prevent (tier-3 never decides alone; human spot-check at a frozen rate); you now have the number for what happens otherwise: 91 % wrong on 699 rollouts. (2) **Practice is not a test**: your own resets (undo primitives, scripted layouts) will under-sample hard states; the robot-day runbook needs operator-staged layouts drawn from a distribution the harness never produced, and the register should record who placed the objects. (3) **Best-of-3 as the fairness control** for any "Claude Code improved X over the week" story: did it learn, or did it try more? Log attempts and budget beside every improvement row. And the SO-101 line you can say: your tools (chapter 15.3) are a code-as-policy agent with a library (`journal/`, the register); SEAR's Fig. 6 says that library helps most when the base success is 30–80 %, which is where the bowl task sits.

---

## 14a.4 Robo-COP (Zhang, Wang, Bauer, Yu, Huang, Wu, Pavone; Stanford / NVIDIA; 2610.09228)

### Method

An orchestrator (Gemini 3.8 Flash writing programs over scripted primitives, perception and an `execute_vla` tool on π0.5) and the policy **co-evolve** in deployment, no exploration phase. Three mechanisms. (1) **Action unification**: scripted and VLA steps are logged as one action stream so a segment that mixes both can become a VLA demonstration labelled "grasp the red block". (2) **Agentic data curation**: a text judge (code, console, feedback) names each segment's intended skill and outcome; a video judge checks sampled frames; a segment enters D only if the text judge accepts *and* the video judge confirms — so a correct pickup from an episode that later failed is kept (Fig. 3). (3) **Policy verification before adoption** (eq. 1): a candidate π′ runs for a window W with the incumbent π as fallback; adopt only if succ(W) ≥ b − δ_task and, for every admitted skill s with enough calls, r_c(s) ≥ r_inc(s) − δ_skill; too few calls → STARVED (discard, memory unchanged); any failure → REVERT. After adoption or reversion, **memory that steered the orchestrator away from the VLA is marked unverified for the current checkpoint**, not deleted, so it can be retested. A controller decides TRAIN or WAIT from curated data and recent failures, naming up to four target skills.

### Protocol and numbers

- Ten RoboLab tasks chosen *before* running, spanning three regimes (VLA alone suffices; scripted primitives suffice; neither). 100 deployment trials per task, first 30 shared across methods before any training; **held-out success on 50 shared initialisations per task**, 95 % beta-posterior credible intervals (Fig. 5). Baselines: π0.5 only; frozen-policy harness (same memory adaptation, no weight updates); fixed schedule (train every 30 trials, deploy every checkpoint, no verification); ASPIRE adapted.
- Fig. 5: mean held-out 28.0 (π0.5) / 54.0 (ASPIRE) / 64.8 (frozen) / 65.8 (fixed) / **73.8 (Robo-COP)**; fixed-schedule training recovers only 1.0 of the 9.0-point gain — training without verification is nearly worthless. Gains concentrate on the six tasks where a candidate was adopted (+11.3 over frozen) vs +5.5 where it was not. Table 1 by regime: VLA-alone 89.3 / 89.3 / 80.0 (frozen / fixed / ours — ours slightly *worse*, honestly reported); primitives-alone ASPIRE 73.5 leads; neither 53.3 / 63.3 / **70.0**.
- Q2: verification rejected 5 of 12 candidates; on the four tasks whose first candidate was reverted, the fixed schedule fell below the frozen baseline on three. Grasp-or-drop accounts for 28 of the 45 fewer failures (131 vs 176 of 500).
- Q3: deployment success 73.2 % vs 68.7 frozen vs 70.3 fixed — learning did not cost deployment success.
- Real (DROID Franka, three tasks, 50 learning trials + 20 held-out, W = 10): 60-s task 35 → 70 % after one verified adoption; 40-s task candidate reverted, 40 → 40 (learning 44 vs 46); 90-s task never trained, 40 → 40. Mean 38.3 → 50.0.

### What is not there

- Intervals on the real-robot numbers (20 held-out per task: 35 → 70 is 7 → 14 successes).
- The judges' own error rates (text and video) are not reported; the curation is trusted by construction, which is exactly what SEAR §3.3 warns about.
- Cost: LoRA fine-tunes during deployment pause the task; the wall-clock is in an appendix.

### For the SO-101

This is G24 (policy replaces scripted executor) written out as an algorithm, and its two numbers decide the design: **fixed-schedule training without verification gained 1.0 of 9.0 points**, and verification rejected 5 of 12 candidates. So the adoption window with fallback and non-inferiority per skill (eq. 1) is not optional, and the thresholds δ_task, δ_skill must be pre-registered. The skill-level curation from failed episodes is also the right reading of AGENTS.md rule 1.2: "failures never train the policy" should become "failed *segments* never train the policy; verified successful segments from failed episodes may" — but only with a video judge whose report card you have (chapter 13.1.1), because Robo-COP never reports one.

---

## 14a.5 Mulligan (Ankile, Dong, Bhowmik, Muppidi, Yuan, Song, Finn; Stanford; 2610.05882)

### The observation and the method

Failures concentrate: 95 % of a policy's failures fall in 25 % of the initial states at 86 % overall success (Fig. 1). Uniform collection therefore spends most operator time on states the policy already handles. Mulligan chooses each round's M supervised initial states (Alg. 2): **failure promotion** (up to m_f states where the deployed policy failed in this round's evaluation, earliest-stage failures first) then **coverage fill** (farthest-point selection in normalised pose space, yaw periodic, from everything collected so far); round 0 uses a scrambled Sobol sequence instead of i.i.d. uniform. **Counterfactual demonstrations**: when a correction reveals the policy's behaviour was wrong well before the intervention, the operator resets to the same initial state and demonstrates the whole task, so the actor's support contains what should have happened. Learners: HG-DAgger (BC on human data) and **HiL-IDQL** (BC actor on D_h; IQL critic trained on D_all including failures; best-of-N chunk re-ranking at deployment, eq. 1). Critic details (eqs. 2–3): h-step chunked bootstrap, λ-return, distributional value head, state-conditioned affine action rescaling.

### Protocol and numbers

- Real: DROID Franka, Insert Marker, Thread Nut, Route Cable, initial states randomised over 15–46 cm ranges; round 0 = 100 demos, each round M = 50 supervised episodes and K = 50 **blinded, interleaved** evaluation episodes on fresh Sobol states shared across methods; **2,550 held-out episodes** in total; Wilson one-SE whiskers; paired per-state tests.
- Fig. 5 / §4.1: final-round HiL-IDQL+Mulligan vs HG-DAgger +34 / +10 / +16 pp; pooled over rounds p < 0.005 on two tasks, 0.04 on the third; Route Cable clip success 37 → 50 %. Simulation (Square-Narrow / Broad, five seeds): HiL methods beat every autonomous method (self-imitation, filtered BC, batch-online IDQL, RLPD, HiL-SERL); RLPD never passes 90 %; HiL-SERL needs > 10× the frames.
- §4.2 ablation (from the abstract and figure): folding autonomous successes into the actor's BC data *lowers* success; the critic on all data including failures is what makes them useful.
- Productivity: with the operator intervening, 98 % of collection episodes complete; HiL-IDQL+Mulligan completes more tasks per robot minute than HG-DAgger.

### What is not there

- The operator placing objects *is* the method; the paper does not test a harness-placed version (your case) — "Mulligan only decides what each round collects."
- Per-round n is 50 and the whiskers are one SE, not 95 %.

### For the SO-101

This is v1.5 §2 ("failed layouts first, then farthest-point coverage") with a published +10–34 pp and the two details you were missing. (1) **Sobol round 0**: your 69 rollouts were not placed by a low-discrepancy sequence; a Sobol layout set for the cube task is a one-hour change to the runbook and removes a confound from every later comparison. (2) **The counterfactual demonstration rule**: when a correction shows the policy was wrong from the start, re-demonstrate from the same layout; that is the same-layout pairing your leader arm makes cheap. And one warning from the ablation: do not fold the harness's own successful rollouts into SmolVLA's BC set (AGENTS.md rule 1.2 is right for a stronger reason than you wrote); they belong to a critic, which you do not have, so for now they belong to the world model and the judge only.

---

## 14a.6 ReTrace, COAP/ETM, HyMeS, residual-foresight RL, HIR-ResRL

Read in the earlier pass; the cards here keep what matters for the register.

**ReTrace (2610.15814).** Rewind-and-retry: on a detected failure, return the arm to a recorded pre-failure state and re-execute, logging the (failure, rewind, retry) triple as recovery data. The usable part on your hardware is the rewind primitive itself — a scripted retreat to the last at-rest pose with the object relation re-verified — which is Spotter's "retreat to recorded pose" and the only recovery primitive you can build this month. The method result to remember: recovery data collected this way is on the policy's own failure distribution, which Mulligan's ablation says belongs in a critic, not the BC set.

**COAP / Experience-to-Memory (2610.12369).** Operator corrections are not trained into the policy; they are stored as retrievable text + keyframe memory and injected into a frozen VLA's context at matching states. The claim is that corrections become usable without a fine-tune, with the same no-regression property Robo-COP buys with a verification window. For you it is the *lightest* adoption path for the tidy-loop corrections: a retrieval table keyed on layout and stage, consumed by the harness's instruction field, tested against "no memory" on paired layouts.

**HyMeS (Weng et al.; 2608.09410).** Memory as an executable program (plan, stage, bindings, counters) written by a coding agent from rollout feedback and injected into a frozen flow-matching VLA as a differentiable constraint on the denoising velocity (eq. 12: v̂ = v + λ_t ∇R, with λ_t decaying as the constraint residual falls, eq. 13). Stage completion by **PACE**: proprioceptive event detection OR a multi-frame VLM (Qwen3-VL-8B) judgment, accepted when k = 3 of the last w = 5 queries agree (eqs. 14–15). Protocol: RoboMemArena 12 tasks / 160 episodes; π0.5 vs PrediMem vs HyMeS, same fine-tuned weights; **SO-101 real, three memory-dependent tasks, 35 trials total**: Observe-and-pick-up 2/10 → 7/10, Number-guided button pressing 7/15 → 11/15, Put-back 0/10 → 2/10; TSR 25.7 → 57.1 %. Table 1 sim: CSR 52.5 → 66.2, TSR 41.3 → 60.1; occlusion is the hardest category and PrediMem falls *below* π0.5 there (38.3 vs 50.4 CSR) — "an unreliable memory can actively mislead execution." Ablations: rollout-refined program vs one-shot +8.5 / +18.4; visual-only verification −16.0 / −21.7; proprio-only −8.8 / −8.4. Not there: intervals (n = 10–15 per real task), a baseline that injects memory without the verification gate. For the SO-101: this is your arm, a frozen VLA, and a code-space memory with stage verification by proprio-OR-VLM voting — the k-of-w vote is a concrete rule for your tier-3 judge ("accept a stage transition when 3 of the last 5 calls agree"), and the occlusion result is the published case of chapter 13's "a judge that decides alone can make things worse".

**Residual-foresight RL (2607.16506).** A residual policy over a frozen VLA, with a latent world model rolled forward to give the critic a foresight feature, for long-horizon tasks. Same family as Imagine-RL (12b.7); the method lesson is the same (critic discrimination as the report card); not runnable on the Orin.

**HIR-ResRL (2606.22860).** Hierarchical residual RL where human interventions supply the reward signal and a high-level selector chooses which residual to apply. Reading only; the one transferable line is that an intervention is a label (Mulligan's "the operator who decides when to intervene also places the objects"), so your key-press-per-episode discipline is already producing the signal these methods consume.

---

## 14a.7 SEVO (Fang, Zhuang, Miao; UConn; 2605.11114)

### What it is

The only paper in the knowledge base that changes the **observation pipeline** of a SO-101 and measures it with hundreds of real trials. Three changes, policy unchanged: body-fixed cameras whose combined field of view covers the whole manipulation (no wrist camera); a 5 W red LED (620–630 nm) at the arm base that physically normalises the appearance of a transparent bottle; a real-time YOLOv8-seg mask overlaid at α = 0.45 in yellow (eq. 2: Ĩ = (1 − αM) ⊙ I + αM ⊙ C), giving a background-invariant cue. Plus a **data-collection protocol**: varied backgrounds, lighting on/off/dark, distractors and people walking through, null episodes (no bottle, arm stays still), and **YOLO–motion binding** (the operator starts moving only once the overlay is visible, so the policy learns to activate on detection). ACT 80 episodes / 400k steps; SmolVLA 120 episodes / 20k steps from the official checkpoint; RTX 4090.

### Protocol and numbers

- Robot A: SO-101 on a cart with an Orin NX, two body-fixed cameras; Robot B: a different cart, three cameras, Raspberry Pi 5 (ACT ~2 min per grasp, SmolVLA ~15 min). Pick-and-place of a transparent water bottle. Three settings: training room; novel-similar rooms (dark floors, different furniture); novel-extreme (white floor). **N = 100 trials per condition.**
- Table II ablation (body-fixed cameras only, averaged over training + similar-novel): full 90 / 79 (ACT / SmolVLA); no YOLO 87 / 76; no red light 84 / 71; **no varied backgrounds 75 / 61**; no light + no BG 66 / 53; baseline raw RGB 63 / 51. Ranking: varied backgrounds > red light > YOLO overlay, consistent across both policies and both robots.
- Table III transfer: with SEVO ACT 95 / 85 / 10 (train / similar / extreme), SmolVLA 83 / 75 / 5; without, 75 / 30 / 0 and 70 / 35 / 0. The extreme drop is explained and closed by collecting 80 episodes in the white room.
- **Table IV, wrist camera added as a third input** (same data): ACT 90 → 86; **SmolVLA 79 → 12**, with every ablation row ≤ 3. Two mechanisms (§VI-H): spatial information poverty at close range (the YOLO mask fills the wrist frame) and motion-induced failures (the wrist view's optical flow during cart motion; ACT oscillates, SmolVLA repeats grasp-release in open air).
- Data floor: ACT ~80 % at 40 episodes, 95 % at 80; SmolVLA nothing below ~80, 83 % at 120 — attributed to the frozen SigLIP encoder (SEVO's overlay reaches ACT's trainable conv1 directly; SmolVLA's action expert must decode it from frozen tokens).

### What is not there

- Intervals (at n = 100 the Wilson 95 % on 90 % is 83–94; most Table II rows are separable, the 3-point YOLO effect is not).
- A mobile-base-stationary control for Table IV (the wrist-camera failure is confounded with cart motion, which the authors name).
- Per-condition success in the extreme room broken down by cause.

### For the SO-101

Your rig *is* Robot A minus the LED and the overlay, and the paper answers three questions you have been carrying. (1) **The wrist camera**: SmolVLA 79 → 12 when it is added, on a SO-101, with the mechanism stated. Your G-series rollouts all use top + wrist; the cheapest experiment in this whole book is **re-evaluating an existing SmolVLA checkpoint with the wrist input blanked**, and then a fine-tune with top + side body-fixed cameras on the same 69 episodes. The §8 lighting angle should be run *after* this, because the camera question dominates it. (2) **Varied backgrounds in collection** is worth 15–18 points before any model change; your bowl episodes were collected on one mat. (3) The data floor (SmolVLA needs ≥ 80–120 episodes to use an observation-space cue; ACT uses it at 40) is a statement about your 69-episode sets: for any observation-space change, ACT is the policy that can show it with the data you have.

---

## 14a.8 PEEK (Zhang, Memmel, Kim, Fox, Thomason, Ramos, Bıyık, Gupta, Li; UW / NVIDIA / USC; 2509.18282)

### Method

Offload "where" and "what" to a VLM so the policy learns only "how". A fine-tuned VILA-1.5-3B maps (instruction, image) to a **2D gripper path** p_t (pixel positions of the end-effector to the end of the sub-trajectory, drawn as a dark-to-light red line) and a set of **task-relevant masking points** m_t (8 %-of-image squares revealed on a black canvas; everything else blacked out). The policy — any image-input policy: ACT, π0, 3DDA — trains and runs on the annotated image. VLM queried every H steps (4–6 s per query on an RTX 3090). Training data for the VLM: 2M VQA pairs from 148k trajectories over 9 embodiments, labelled automatically by tracking a grid of points with a video point tracker, keeping points that move (task-relevant entities) and the gripper track, segmented at manipulation boundaries; plus RoboPoint and LIBERO-90.

### Protocol and numbers

- Franka sim-to-real: 2.5k MuJoCo cube-stacking trajectories; real Franka + Zed 2 stereo; 3DDA base policy trained only in sim; Basic (4 fixed configs), Clutter, Semantic (unseen objects, celebrity photos); **5 trials per task, 220 evaluations across 4 methods × 11 variations.** WidowX BRIDGE: BRIDGE-v2 relabelled with PEEK, zero-shot on a custom setup; ACT, π0, OTTER, ARRO baselines; 5 evals per task.
- Fig. 5: Franka overall success 3DDA 0.02 → +PEEK 0.63 (41.4×; +HAMSTER 0.31, +ARRO 0.06); Clutter 0.00 for every baseline, 0.60 for PEEK. WidowX overall: ACT 0.08 → 0.24 (+PEEK), π0 0.16 → 0.69 (4.5×); OTTER 0.07; ARRO does not help either base.
- Table I (sim cube stacking, ablation): base 33.5 ± 3.1; paths only 52.8; masks only 65.6; both 73.6 ± 3.9. "Masks outperform paths since they simplify the scene by removing the distractor cube, while paths alone leave ambiguity."
- VLM design: 13B → 3B costs nothing (DTW 0.12, IoU 0.68); joint path+mask prediction +19.3 % relative on IoU over mask-only.

### What is not there

- n = 5 per task is the lowest in the book; the headline multipliers are ratios of small counts (0.02 → 0.63 overall is roughly 1 → 35 successes out of 55).
- Latency: a 4–6 s VLM query every H steps rules out anything dynamic; H is in the appendix.
- A non-VLM mask baseline that uses the *correct* object (an oracle mask) to separate "the VLM finds the right object" from "masking helps".

### For the SO-101

A VLM at 4–6 s per query is at rest only on your hardware, and the thing to take is the **ablation shape**, not the system: masks (where) > paths (what), and both > either. Your G9c-3 judge already has the HSV masks for the bowls; the one-evening test is **training ACT on the 69 episodes with the judge's mask overlaid** (SEVO's overlay with your own detector instead of YOLO), paired against plain RGB on the same layouts. If it moves the wrong-object failure rate (fact 7), you have a result on the identity problem that no VLA paper in the base addresses; if not, you have ruled out the cheapest perception fix before buying a camera.

---

## 14a.9 Blind precise assembly (Bronars, Park, Kedia, Agrawal; MIT; 2610.23849)

### What it is

Peg insertion, gear meshing and nut threading from **joint positions, velocities and previous actions only** — no image, no force/torque sensor, fixture pose and grasp unknown. PPO in simulation over thousands of procedurally generated assemblies (`mjfit`: exact convex decomposition that preserves 0.5–1.5 mm clearances, Figs. 4–6), 12,288 parallel worlds; a temporal convolution over the last 0.6 s (H = 30) → latent z_t → MLP → joint-position increments for an impedance controller. Training-only aids: an asymmetric critic with privileged pose, and a **contact decoder** predicting the contact state from z_t (cross-entropy, training only, dropped at deployment, Fig. 7) — the actor never sees anything privileged.

### Protocol and numbers

- Sim: all three tasks near-perfect in distribution (Fig. 8, two seeds). Fig. 9: blind policy seats later than a visual policy (1.42 vs 0.32 s) but travels a median 37 mm vs 95 mm for an optimistic scripted spiral, over 3,964 matched starts. Fig. 10: a linear probe on z_t decodes the task mode (lifted / near / partial / moving / stuck …) at 61–73 % vs 41–64 % from the current observation, chance 14 %; the same probe fitted in sim decodes the same sequence on the real robot.
- Fig. 11: on unseen parts (tighter, slender, large, random convex) the blind policy degrades far less than the matched visual policy (wrist RGB 64×64, same everything else).
- Real: Flexiv Rizon 4S + Robotiq 2F-85 at 50 Hz, zero-shot; eight 3D-printed pegs, five in distribution and three out (0.5 mm clearance, larger, taller handle); **237 trials, 173 successes (73 %); per-peg 22–24 / 30 in distribution, 19 / 29, 19 / 29, 21 / 29 out; 95 % Wilson intervals on every bar (Fig. 12)**, beside the same policy's simulation score on the same peg (2,048 episodes).

### What is not there

- The arm is torque-controlled with an identified impedance model; the method depends on compliant tracking of joint-position increments, which a STS3215 servo bus does not provide.
- Only the peg policy was transferred; gears and nuts are sim-only.
- No comparison to a force-sensing baseline on the real robot.

### For the SO-101

Not a method you can run, but the cleanest real-robot evaluation in the entire knowledge base: 237 trials, Wilson intervals on every bar, the same policy's simulation score beside each real bar, in- and out-of-distribution parts labelled, and a mechanism probe (Fig. 10) that is fitted in sim and checked on hardware. It is the standard to point at when someone asks what a real-robot result should look like, and chapter 7's Wilson exercise is this figure. One idea does transfer: the **contact decoder as a training-only auxiliary on a history latent** is the same pattern as PSG-JEPA's grounding heads (12b.4) and your W1 probes; and the probe result — the history latent decodes contact mode better than the current observation — is the premise behind REKI's memory (11a.4) stated from the proprioceptive side. Your arm reads joint positions and goal positions; the difference between them under load is a crude contact signal you have never logged. Log it.

---

## 14a.10 How this family evaluates, side by side

| Paper | Platform | Real n | Intervals | Controls for "looks like learning" | Verifier error reported |
|---|---|---|---|---|---|
| RegenHarness | quadruped deployment | — | — | — | — |
| SEAR | 330 sim tasks; 5 physical | 10 staged scenes/task | 3-run spread (sim) | **best-of-3, sealed judge, 5 tiers, 3 repetitions, matched budget** | yes (91 % wrong verifier, §3.3) |
| Robo-COP | 10 RoboLab; 3 DROID | 20 held-out/task | 95 % credible (sim) | frozen-policy harness; fixed schedule without verification | no |
| Mulligan | 3 DROID real; 2 sim | 50 blinded/round, 2,550 total | Wilson 1 SE; paired tests | uniform sampling at matched budget; autonomous methods | n/a (operator labels) |
| HyMeS | RoboMemArena; **SO-101** | 10–15/task, 35 total | none | same fine-tuned weights, memory on/off | partial (ablation of PACE signals) |
| SEVO | **SO-101 ×2** | **100/condition** | none (computable) | full ablation grid; two robots; two policies | n/a |
| PEEK | Franka, WidowX | **5/task** | ± on sim ablation only | base policy; 2 mask baselines | no |
| Blind assembly | Flexiv | **237 total** | **Wilson 95 % every bar** | matched visual policy; scripted spiral; sim beside real | n/a |

Read with chapters 3–7:

- The **best real-robot protocol** in the knowledge base is a proprioception paper (blind assembly) and the second best is a data-collection paper (Mulligan: blinded, interleaved, paired, Sobol). Neither is a world-model or VLA paper. When asked "who evaluates well", name those two.
- The **best controls for self-improvement** are SEAR's five, and they transfer directly to the flywheel: log attempts and budget, hold out layouts the harness never produced, keep the judge sealed from the agent, repeat, and match time.
- Verifier error is reported in exactly one paper (SEAR, by accident: a verifier that was 91 % wrong). Robo-COP's whole curation pipeline rests on two judges whose error is never measured. Your report cards are ahead of the field here; the thing to add is SEAR's warning, in the rule: **a verifier never serves as the stopping rule for the loop that calibrated it.**
- n = 5 (PEEK) and n = 10 (SEAR hardware, HyMeS) appear in 2026 papers from strong groups. The field's hardware n is small because hardware time is expensive, not because small n is fine; say the Wilson interval when you cite them.

## 14a.11 Corrections and additions to chapters 13–14, v1.5 and the runbook

- v1.5 verdict vocabulary: adopt RegenHarness's five words; map G9c-3 "invalid" → *ambiguous*; add *stale* for any verdict computed on a frame older than the last action. Role-isolated verifier prompt as a tested rule.
- Harness configuration changes (thresholds, budgets, prompts) go through a frozen regression set of recorded trials before going live (RegenHarness eq. 7).
- G24 adoption: pre-register δ_task and δ_skill; STARVED / REVERT / ADOPT as the three outcomes; mark policy-dependent memory unverified on every switch (Robo-COP §3.3). Expect to reject ~40 % of candidates.
- AGENTS.md rule 1.2 refined: failed *segments* never train the policy; successful segments from failed episodes may, only through a judge with a report card; harness successes never enter the BC set (Mulligan ablation) — they go to the world model, the judge and (later) a critic.
- Runbook: Sobol layout set for round 0; operator-staged (not harness-reset) layouts for every held-out block; record who placed the objects; counterfactual re-demonstration from the same layout when a correction shows the policy was wrong from the start.
- Flywheel claims: log attempts, budget and restarts beside every improvement row (SEAR Table 2); the judge that calibrates a loop is never its stopping rule.
- Chapter 10 / camera: the wrist camera is now a pre-registered question, not a default — SmolVLA 79 → 12 on a SO-101 (SEVO Table IV). First experiment: re-evaluate the existing checkpoint with the wrist blanked; second: fine-tune top + side. §8 lighting runs after.
- Chapter 13.3 data: varied-background collection is worth 15–18 points on this arm before any model change (SEVO Table II); the next bowl/cube session varies the mat and the lighting by design.
- New cheap gate: ACT on the 69 episodes with the G9c-3 mask overlaid vs plain RGB, paired layouts, judged on wrong-object rate (PEEK Table I shape, SEVO overlay mechanism).
- Log the commanded-vs-measured joint gap on every frame; it is the only contact signal this arm has (blind assembly Fig. 10).
- Tier-3 stage transitions: accept when k = 3 of the last w = 5 verifier calls agree (HyMeS eqs. 14–15); never let a memory or stage advance on a single call.

## 14a.12 Exercises

1. Re-score G9c-3's existing verdicts with the five-word vocabulary; count *ambiguous* and *stale* separately; say what each would have changed in the register.
2. Build SEAR's Table 2 for your repository: for each of the five "looks like learning" rows, write the control you have, the one you lack, and the cost of adding it.
3. Write the G24 adoption test as eq. 1 of Robo-COP with your own δ_task and δ_skill; justify both from chapter 7's power calculation at the robot-day n.
4. Generate a 50-layout Sobol set for the cube task in the workspace coordinates; plot it against the 69 recorded layouts; compute the farthest-point gap.
5. Re-evaluate the current SmolVLA checkpoint on the 69 episodes with the wrist camera input blanked (offline, paired); report the change in the G9c-3 verdict distribution. Then write the top+side fine-tune protocol with its bar.
6. Write the Wilson interval beside every real-robot number cited in this chapter (SEAR Table 4, HyMeS Table 2, PEEK Fig. 5, Robo-COP §4.3, SEVO Table II); then beside every number in your own register that lacks one.
