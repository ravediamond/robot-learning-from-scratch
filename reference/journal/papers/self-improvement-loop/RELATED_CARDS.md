# Self-improvement loop — related work cards (WS-083, self-improvement item, step 1)

Read in full from the papers mirror after `papers-sync` (2026-11-01): ENPIRE, LoopWAM, HALTER, the three marked
"full" in `RELATED.md`. The other 13 PDFs of RELATED.md's list are **not on Drive yet** (listed at the end); their
entries in FEASIBILITY.md §4 stay abstract-level and say so. Card shape as WS-081: claim with its table, their rig,
what transfers to an SO-101 + Orin and what does not, the cheapest version of their mechanism here, cost.

## Numbers in RELATED.md checked against the PDFs

| RELATED.md says | PDF | status |
|---|---|---|
| ENPIRE: 8 bimanual YAM stations, one RTX 5090 each; Codex, Claude Code, Kimi | Fig. 1, Table 1, §3 | correct |
| ENPIRE: verifier from success / failure recordings, < 150 ms | §2.1: "a few minutes of success and failure demonstrations", "under 150ms" | correct ("a handful" → "a few minutes of") |
| ENPIRE: safety violation = failure + reset; reset by SAM3, pose tracking, cuRobo, torque-verified grasp | §2.1; appendix (SAM3, BundleSDF, cuRobo, gripper torque) | correct |
| ENPIRE: up to 99 % on pin insertion / zip-tie | abstract: "a 99% success rate" on tasks such as Push-T, pins, zip tie; pin insertion run to 50 consecutive successes | correct |
| ENPIRE: 2 of 3 agents failed real Push-T after solving it in sim | §3: all solve sim; real "causing two of the three agents to fail" | correct |
| ENPIRE: 1 → 8 robots cut wall-clock 2–3×, token cost super-linear | §3.3: Push-T ~5 h → ~2 h; pins > 1.5 h → ~40 min; §4 "Token cost grows super-linearly" | correct |
| LoopWAM: 3 rounds, 4 embodiments, 30 tasks, 23.7 → 28.0 → 37.7 → 43.3 % | §4: RoboChallenge Table30-V2, ALOHA / DOS-W1 / ARX5 / UR5, 23.67 → 28.00 → 37.67 → 43.33 % | correct |
| LoopWAM: quality token, source-dependent noise, λ(q) non-decreasing, λ = 0 on failures with video kept; coarse annotation one score per checkpoint | §3.2 | correct |
| LoopWAM: "expert + rollouts from other policies also works" | §3.3 / §4.3: validated on RoboTwin 2.0 and RoboCasa365 | **corrected**: simulation only |
| HALTER: 6 reset skills per scene, 30–50 demos each, X-VLA; RGB-D + GroundedSAM; Qwen3.5-397B | §IV-A | correct |
| HALTER: reset SR 76 % vs 52 % AutoEval; interventions 100 → 25; operator time −72 %; held-out 74.7 % | Tables I–III (76.0 ± 4.3 vs 52.0 ± 5.0; 100 → 25; 88 → 24 min; 74.7 ± 5.0 vs 1.3 ± 1.3) | correct |
| RELATED.md §"What this changes" 3: "HALTER's numbers say a learned reset of a single step reaches 85–95 %" | not in the PDF; per-task HALTER reset SR is 88 / 84 / 48 / 84 % on multi-step tasks (Table I) | **corrected**: removed the 85–95 % figure |

---

### 2606.19980 · ENPIRE (agentic robot policy self-improvement in the real world)
- **Claim:** coding agents, given an environment of immutable APIs (hard safety limits, an automated binary verifier, automated reset) built first with human feedback, improve real policies autonomously — up to a 99 % success rate on Push-T, pin insertion, zip-tie cutting (abstract; Fig. 3); scaling 1 → 8 robots cuts time to success (Push-T ~5 h → ~2 h, pins > 1.5 h → ~40 min, §3.3) at super-linear token cost (Fig. 7).
- **Their rig:** 8 bimanual 6-DoF YAM stations, each with its own RTX 5090 workstation and coding agent (Codex, Claude Code, Kimi Code); SAM3, BundleSDF, cuRobo for reset.
- **Transfers:** the order (environment first, frozen as an API; then the loop), and the three artefacts — this rig has the verifier (G9c-3, 0 FP / 88), needs the safety clamp (the watchdog) and the reset.
- **Does not transfer:** SAM3 + BundleSDF + cuRobo beside the policy on a 16 GB Orin; 8 stations; a coding agent as the improver (here the improver is a fixed fine-tune recipe).
- **Cheapest version here:** `env.py` with three frozen calls — `clamp()` (joint / workspace limits, stall timeout), `judge()` (G9c-3 with its end-rule), `reset()` (manual in the pilot) — committed before rollout 1, hash in the pilot log.
- **Cost:** ~2 Orin days (watchdog + end-rule) + 1 arm day of watchdog tests; $0.

### 2610.18873 · LoopWAM (closed-loop policy improvement from deployment rollouts)
- **Claim:** one world–action model trained on *all* rollouts — successes, suboptimal and failed — with a quality score in the instruction, source-dependent noise and action loss weighted by λ(q) (λ = 0 for failures, video kept) improves over 3 deployment rounds from 23.67 to 43.33 % on 30 real tasks over 4 embodiments (§4, RoboChallenge Table30-V2); coarse annotation needs one score per checkpoint, failures get the minimum (§3.2).
- **Their rig:** ALOHA, DOS-W1, ARX5, UR5; a dual-stream diffusion transformer from Wan2.2-TI2V-5B (multi-GPU training).
- **Transfers:** the quality suffix on the instruction and the masked action loss for failed episodes — a few lines in a SmolVLA fine-tune; coarse annotation is exactly what a binary verifier gives.
- **Does not transfer:** the video stream (SmolVLA has no world-model head), so failed rollouts carry **no** training signal here unless something else reads them: in SmolVLA with λ = 0 a failed episode contributes no loss at all. The honest ablation here is "quality token on successes + iteration-0 demos" vs "successes only", not LoopWAM's full mechanism. Mixed-policy rollouts: shown in simulation only.
- **Cheapest version here:** arm 1b of the pilot — same 40 rollouts, instruction suffix "Quality: q" (demos q*, verified successes q_s), failed episodes kept with action loss masked (λ = 0), one more ~$3 fine-tune, same held-out 20.
- **Cost:** ~$3 vast, + 20 held-out trials (~0.5 arm day).

### 2609.19413 · HALTER (graph-based harness for autonomous long-horizon evaluation)
- **Claim:** reset as planning over a small library of atomic reset skills (6 per scene, X-VLA, 30–50 demos each), with an online scene graph (relations on / in) read by an LLM that scores the rollout, plans the reset and verifies it: reset SR 76.0 % vs AutoEval 52.0 % and MP skills 65.0 %, verification 91.0 % (Table I, 4 tasks × 25 episodes, Franka); interventions 100 → 25, operator time 88 → 24 min (Table II); 3 held-out tasks reset 74.7 % vs 1.3 % with no new demos (Table III).
- **Their rig:** Franka Panda 7-DoF, three RGB-D cameras, GroundedSAM, Qwen3.5-397B.
- **Transfers:** reset skills as a separate small library composed per terminal state; a verifier that reads relations, not pixels — the bowls scene has 2–3 relations (blue in pink, pink on mark, blue on mark), and G9c-3's masks already read "in".
- **Does not transfer:** depth (none here), a 397B LLM, the 7-DoF reach; the hardest task's reset was still 48 % (Pot-Cook), so a learned reset is not reliable by default.
- **Cheapest version here:** 2 reset skills (bowl back to its mark; blue bowl out of pink), 30 teleop demos each, SmolVLA recipe, chosen by the G9c masks' relation, verified by the same masks; vs the dump-and-restack fixture.
- **Cost:** ~1 teleop session (60 demos) + ~$3 fine-tune + 1 arm day of reset trials; fixture: parts < €50, 3–5 days.

---

## Not on Drive yet (13 of the 16 ids in RELATED.md)

2607.14047 Zero2Skill · 2609.32069 Find Something You Can't Do · 2607.26809 Practice Makes Policies · 2609.38178
Skill-Space Shooting · 2610.02204 Reconstruct, Practice, Go Real · 2608.11363 What Does It Take … Minimal Data ·
2609.21788 Subtask RL · 2610.13345 Evo-RL · 2607.01111 FAR · 2609.14633 REVOLVE · 2610.03476 MobiAgent · 2609.38905
EmbodiRSI · 2603.11811 RADAR. Also cited and not read: AutoEval (CoRL 2025). RELATED.md names 17 papers; it lists
16 arXiv ids plus AutoEval, which has none.

---

## Added 2026-11-01 (night run): two execution-error papers named by the owner with this item

`papers-sync` on 2026-11-01 found 2 new files (2609.37334, 2609.33145); the 13 loop PDFs above are still not on Drive.
Neither new paper is a self-improvement loop; both are about the gap between what the robot is commanded and what it
does, which bears on how the loop's training targets are built (and on WS-082 C1 / C2).

### 2609.33145 · CSDW (Beyond State-as-Action: command–state discrepancy)
- **Claim:** training on recorded commands instead of measured states, weighted by the command–state discrepancy, raises full-insertion success: cable insertion State 25.7 % → Command 60.0 % → CSDW 94.3 % (35 trials); connector mating 0.0 → 33.3 → 73.3 % (30); package placement 80.0 / 80.0 / 83.3 % (Table I).
- **Their rig:** Marvin dual-arm robot, VR teleoperation, π0.5; 482 / 1,044 / 151 demos.
- **Transfers:** LeRobot SO-101 data already records both sides — `action` is the leader's position (the command), `observation.state` the follower's (the state). The loop's fine-tune uses `action` as targets, i.e. Command supervision; CSDW's weights are computable offline from the same two streams.
- **Does not transfer:** VR teleop with an impedance controller (the SO-101 follower is position-controlled); contact-rich insertion (the bowls task has little sustained contact).
- **Cheapest version here:** compute the leader–follower discrepancy on the 182 bowls demos; if it is concentrated at the grasp, CSDW weighting is a one-line change in the fine-tune loss — a candidate second ablation, not added to the pilot.
- **Cost:** $0 to measure (offline); ~$3 per weighted fine-tune.

### 2609.37334 · Self-compensating VLA and RoboStress
- **Claim:** updating a VLA online from the residual between commanded and executed motion (no labels, no reward) raises real success by > 30 points on two AgileX Piper arms: average 43 → 76 % (π0.5, new arm), 43 → 74 % (π0.5, one-year-old arm), 25 episodes per task (Table 6); even the new arm has a mean normalised command–execution residual of 32.9 % (35.0 % on the old one). In simulation (RoboStress: friction, backlash, compliance, gravity-compensation error) 49.1 → 56.6 % with π0.5 (Table 2).
- **Their rig:** AgileX Piper 6-DoF arms, π0 / π0.5 on a GPU; LIBERO-based simulation.
- **Transfers:** the measurement — the commanded-vs-executed residual per joint is in every SO-101 recording, and STS3215 servos have exactly the backlash and compliance the benchmark models.
- **Does not transfer:** online gradient updates of a VLA during deployment on the Orin (the loop updates offline between iterations).
- **Cheapest version here:** report the normalised residual of the SO-101 on the existing rollouts with the paper's definition, beside the Piper's 32.9–35.0 % (this is WS-082 C1 / C2's first number).
- **Cost:** $0 (offline).
