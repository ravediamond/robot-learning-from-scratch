# Self-improvement loop — related work on letting the robot run the loop itself (read 2026-10-31 evening)

Searched for closed loops with automatic verification and reset, 2025–2026. None in the papers mirror; the owner
adds the PDFs to Drive. Read in full: ENPIRE, LoopWAM, HALTER. Abstract-level: the rest.

## The three that matter

**ENPIRE (2606.19980, NVIDIA / CMU / Berkeley, Jun 2026).** Coding agents (Codex, Claude Code, Kimi) run the whole
loop on 8 bimanual YAM stations, RTX 5090 each. The design is two stages and the first is the one to copy: a
human-guided *environment construction* phase where the agent writes (1) hard safety constraints — joint-space and
kinematic limits, a violation = failure + reset; (2) an automated binary verifier from a handful of success / failure
recordings ("a few minutes" of them, checked 2026-11-01), optimised for accuracy and < 150 ms; (3) an automated reset as procedural tool calls (SAM3 detection,
pose tracking, cuRobo planning, torque-verified grasp). Only then the agent is let loose on policy improvement
(BC, offline / online RL, hyperparameters) with the environment exposed as immutable Gym APIs. Results: up to 99 %
on pin insertion / zip-tie; 2 of 3 agents failed real Push-T after solving it in sim. Scaling 1 → 8 robots cut
wall-clock 2–3× but token cost grew super-linearly. **What transfers:** the split (freeze the environment, then
improve) and the three artefacts; the verifier here already exists (G9c-3), the safety clamp is the watchdog to build,
the reset is the open item. **What does not:** cuRobo / SAM3 on an Orin beside the policy; 8 stations.

**LoopWAM (2610.18873, Tsinghua / Dexmal / Loopeva, Oct 2026).** The answer to "training on your own successes
only narrows the policy": do not discard failures. One world–action model trains on *all* deployment rollouts —
success, suboptimal, failed — with (a) a quality score as a conditioning token (expert q* > any rollout; at
deployment condition on q*), (b) source-dependent noise, (c) action-loss weight λ(q) non-decreasing in quality,
λ = 0 for failures but their video supervision kept. Coarse annotation suffices: one score per checkpoint, successes
get it, failures get the minimum — i.e. a binary verifier is enough. Real result: 3 rounds on 4 embodiments, 30 tasks,
23.7 → 28.0 → 37.7 → 43.3 % success; expert + rollouts from other policies also works *(corrected 2026-11-01 against the PDF: shown in simulation only, RoboTwin 2.0 and RoboCasa365)*. **What transfers:** the
quality token and λ(q) weighting are a few lines in a SmolVLA fine-tune (condition on a quality string, mask the
action loss of failed episodes) — it lets failed rollouts into the training set without violating "never train
on failed rollouts" in the sense that matters (their actions are never imitated). This is the cheapest upgrade to
the pilot's "successes only" recipe and a clean one-factor ablation. **What does not:** the video-prediction head
(WAM) is not in SmolVLA; the ablation here is quality-token + masking only.

**HALTER (2609.19413, KIT / UNC, Sep 2026).** Reset for long-horizon tasks as planning over a library of atomic
reset skills (6 per scene, 30–50 demos each, X-VLA), with a scene graph built online from RGB-D + GroundedSAM
and an LLM scoring the rollout, planning the reset and verifying it. Franka: reset SR 76 % vs 52 % AutoEval, human
interventions 100 → 25 per 100 episodes, operator time −72 %; held-out tasks reset at 74.7 % with no new demos.
**What transfers:** the idea that the reset skills are a *separate* small library that the LLM composes, and that the
verifier reads relations (on / in) not pixels — the bowls / cubes scene has 3–4 relations. **What does not:** RGB-D
(no depth here), Qwen3.5-397B, a 7-DoF reach. For the SO-101 the honest options stay: a dump-and-restack fixture
(mechanical), or 2–3 reset demos per terminal state taught as skills (bowl back to mark, cube out of bowl), with the
G9c masks as the relation reader.

## The rest, one line each

- Zero2Skill (2607.14047): autonomous collect → train → deploy with self-resetting and VLM verification; the
  pipeline shape, large hardware.
- Find Something You Can't Do (2609.32069, TU Darmstadt): agentic real-world RL — the agent *searches for* the
  failures of a frozen VLA and trains on them; the layout-chooser idea in the harness v1.5, done as RL.
- Practice Makes Policies (2607.26809): bootstrapping from zero human demos; the opposite starting point.
- Skill-Space Shooting (2609.38178, Tsinghua / Berkeley): improvement by sampling in a skill latent, no retrain
  per iteration.
- Reconstruct, Practice, Go Real (2610.02204, Berkeley / Amazon FAR / MIT): reconstruct the scene, practise in
  sim, deploy; needs a sim of the scene.
- What Does It Take … Minimal Data (2608.11363, CMU): the data-efficiency question for generalists adapting from
  autonomous interaction; read for the ablation design.
- Subtask RL (2609.21788, UT Austin): RL only on the failing subtask of a long-horizon task; the WS-062 angle.
- Evo-RL (2610.13345): selective experience — which rollouts to keep; LoopWAM's question with RL.
- FAR (2607.01111, CMU): failure-aware retry at test time + continual improvement; the retry is what SaiVLA-JEPA
  also found works.
- REVOLVE (2609.14633), MobiAgent (2610.03476), EmbodiRSI (2609.38905): closed-loop frameworks with minimal
  human intervention; abstract-level only.

## What this changes in PILOT.md and FEASIBILITY.md §3

1. The pilot gains a second arm, same data, same cost: iteration 1a = successes only (the frozen recipe); iteration
   1b = LoopWAM-style, all 40 rollouts with a quality token and λ = 0 on failed actions. One fine-tune each (~$3
   each). The bar stays on the held-out 20; the comparison 1a vs 1b is the paper's first figure even if neither beats
   iteration 0. AGENTS.md §1 ("never train on failed rollouts") is kept as written for 1a and must be read as "never
   imitate failed actions" for 1b — the owner decides whether to amend the rule or keep 1b exploratory.
2. ENPIRE's order is adopted: environment first (watchdog, verifier end-rule, reset), frozen and committed as an API,
   then the loop. The full loop is not started until the three exist; the pilot runs with manual reset but with the
   watchdog on.
3. Reset: cost the two SO-101 options against each other before building either — fixture (parts < €50, 3–5 days)
   vs taught reset skills (2 × 30 demos, one teleop session, reuses the SmolVLA recipe, verified by G9c masks).
   *(Corrected 2026-11-01: the 85–95 % single-step figure is not in HALTER's PDF; its per-task reset SR is 88 / 84 / 48 /
   84 % on multi-step tasks, Table I.)* The bowls reset is one or two steps.
4. The novelty sentence of the paper, after this read: not "a closed loop" (ENPIRE, LoopWAM, Zero2Skill have them)
   but "a closed loop whose verifier, reset and fine-tune together cost under $5 per iteration on a $200 arm and an
   edge GPU, with the LoopWAM weighting as the one-factor ablation" — cost per iteration is the axis none of them
   reports.

## PDFs to add to Drive (`Projects/Robotics/Papers`), then `papers-sync`

| id | title | read here |
|---|---|---|
| 2606.19980 | ENPIRE: Agentic Robot Policy Self-Improvement in the Real World | full |
| 2610.18873 | LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts | full |
| 2609.19413 | From Rollout to Reset: HALTER, a graph-based harness for autonomous long-horizon evaluation | full |
| 2607.14047 | Zero2Skill: Bootstrapping Robot Skills through Autonomous Data Collection, Training, and Deployment | abstract |
| 2609.32069 | Find Something You Can't Do: Agentic Real-World RL for Self-Improving VLA Models | abstract |
| 2607.26809 | Practice Makes Policies: Bootstrapping and Consolidating Robotic Capabilities from Zero Human Demonstrations | abstract |
| 2609.38178 | Skill-Space Shooting for Autonomous Robot Policy Improvement | abstract |
| 2610.02204 | Reconstruct, Practice, Go Real: Guided Self-Improvement for Embodied Agents | abstract |
| 2608.11363 | What Does It Take For A Generalist Robot Policy To Adapt Using Minimal Data? | abstract |
| 2609.21788 | From Pretraining to Proficiency: Real-World Subtask RL for Long-Horizon Manipulation | abstract |
| 2610.13345 | Evo-RL: Iterative Robot Reinforcement Learning from Selective Experience | abstract |
| 2607.01111 | FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement | abstract |
| 2609.14633 | REVOLVE: An Automated Closed-Loop Framework for Evolving Robot Manipulation | abstract |
| 2610.03476 | MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation | abstract |
| 2609.38905 | EmbodiRSI: Recursive Self-Improvement for Data-Efficient Robot Adaptation | abstract |
| 2603.11811 | RADAR: Closed-loop robotic data generation via semantic planning and autonomous causal environment reset (cited by HALTER, not read) | — |
| AutoEval (CoRL 2025, Zhou, Atreya, Tan, Pertsch, Levine) | Autonomous evaluation of generalist robot manipulation policies in the real world (cited by HALTER, not read) | — |
| 2609.37334 | Taming VLAs under Robot Execution Errors: Self-Compensation and Stress Testing (for C2, not the loop) | abstract + §1 |
| 2609.33145 | Beyond State-as-Action: Exploiting Command–State Discrepancy for Robot Imitation Learning (for C1 / C2) | abstract |
