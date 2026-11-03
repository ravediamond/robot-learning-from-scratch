# WS-082 · Ten novel paper candidates (step 1)

Questions this rig can measure that the library's papers did not, for a structural reason. Gap map: `GAPS.md`.
Reading only: nothing was run, and every bar is a **seed** for a protocol the owner freezes (WS-069 template), not a
bar. Excluded subjects (WS-069–074) are not drawn on; re-tests of single papers are in WS-081's `TRIAGE.md`.
Scores 1–5: novelty (structural reason), feasibility here, reuse of existing data, citation value for the thesis,
**risk of null scored so that 5 = a null result is still a paper**. Ranked table at the end.

Spread: VLAs on low-cost hardware 2 (C2, C3) · world models 2 (C4, C5) · data / teleop efficiency 2 (C6, C7) ·
chunking / latency 2 (C1, C10) · instruction following 1 (C8) · memory 1 (C9). Extends a register entry: C4 only
(world models, #1). No arm day beyond the queue: C1, C2, C4, C5 (offline, $0).

---

### C1 · Sub-servo smoothing: do chunk-smoothing corrections reach the joints?
1. **Question:** below what commanded step does an SO-101 joint not move, and what share of a chunk smoother's corrections falls under it?
2. **Why new:** RTC 2506.07339 (inpainting across seams), smooth MP chunking 2610.39451 (73–84 % of boundary jerk removed), AAC 2604.04161 (adaptive horizon), PACT 2610.28019 (async cycles), RACE 2610.05719 (errors at sub-skill transitions). All score smoothness on the **commanded** stream: on a Franka or a YAM the joints follow commands far below a seam's size, so commanded = executed and nobody had to check.
3. **Why this rig can:** STS3215 hobby servos with finite resolution and deadband; gate A of smooth chunking found projection moves the first action by only 0.30–0.32° (README table).
4. **Experiment:** factor: commanded per-step joint change, binned (0.05° bins); baseline: identity (arm follows); statistic: measured/commanded gain per bin, per joint, bootstrap over episodes; seed bar: a dead zone ≥ 0.3° (gain < 0.5) on ≥ 3 of 5 arm joints; then the share of S2/S3 correction energy inside it. n: 69 + 19 rollouts.
5. **Reuses:** rollouts (`ravediamond/so101-bowls-policy-rollouts`, commanded `action` vs recorded `observation.state` at 30 fps), smooth-chunking `results/a_rows.jsonl`. New: the binning script.
6. **Cost:** 0 arm, ~0.5 Orin day, $0, 0 teleop, ~0.5 owner h.
7. **Figure:** measured vs commanded joint step per joint, dead zone shaded, gate A's S2/S3 correction histogram overlaid.
8. **If null (no dead zone):** a short note that SO-101 servos track sub-degree commands, so seam smoothing results on this arm carry to the joints; the register's chunking entries gain a hardware check.
9. **Objection:** "gain depends on load and speed." Answer: stratify by joint velocity and by G15 stage (transport vs grasp); the claim is per stratum.
10. **Venue:** ICRA 2027 workshop; short.
11. **Scores:** N4 · F5 · R5 · C4 · null 4 = **22**.

### C2 · Choosing or tracking: how much of a VLA's failure is the arm not following?
1. **Question:** in failed SmolVLA rollouts, is the commanded-vs-measured joint gap at grasp and place larger than in successful ones, i.e. is part of "policy failure" execution error?
2. **Why new:** SO-101 benchmark 2606.08881 (failure taxonomy: grasp instability, state mismatch, precision misalignment, from video), FACTR 2 2606.12406 (external torque on commodity arms), LoRA on SO-101 2512.11921, MolmoAct2 2605.02881. Failure taxonomies are judged from video on arms whose tracking is not a variable; none splits a failure into "chose wrong" and "did not execute" because their arms do execute.
3. **Why this rig can:** a $200 arm under load (gripper on a bowl rim), both streams recorded at 30 fps, G15 stage labels on the same episodes.
4. **Experiment:** factor: outcome (success / failure); baseline: same stage in successes; statistic: per-stage median |commanded − measured| (gripper and wrist joints), Mann–Whitney, episode bootstrap; seed bar: failures ≥ 2× successes at the grasp stage with CI excluding 1. n: 69 labelled rollouts (46 / 23).
5. **Reuses:** the 69 rollouts, G15 stage labels, G9c-3 outcome labels. New: per-stage alignment script.
6. **Cost:** 0 arm, ~0.5 Orin day, $0, 0 teleop, ~1 owner h (spot-check 10 episodes).
7. **Figure:** per-stage tracking gap, failures vs successes, with the share of failures where the gap exceeds the gripper's rim tolerance.
8. **If null:** failures here are choices, not execution: the register's policy comparisons are not confounded by the arm (a useful negative for every SO-101 paper).
9. **Objection:** "a large gap is a consequence of failure (blocked arm), not a cause." Answer: time order — test the gap in the window before the failure onset (`shared/gate/data/2026-10-23_onsets_vlm/onsets.csv`); report both.
10. **Venue:** RA-L letter (with C1 and C3 as one "execution on a $200 arm" paper) or CoRL 2027 workshop; short.
11. **Scores:** N4 · F5 · R5 · C4 · null 3 = **21**.

### C3 · Servo heat as a hidden variable across a robot session
1. **Question:** over 60 back-to-back rollouts, does servo temperature raise tracking error and lower success?
2. **Why new:** SO-101 benchmark 2606.08881 (20 trials per cell, no temperatures), Real-Time EXPO-FT 2609.18207 (30 trials, industrial arm), FACTR 2 2606.12406, ACT 2304.13705. Their actuators have thermal headroom and trials are interleaved with resets by staff; hobby servos lose torque above ~65 °C (the rig's B8 risk) and a session without a human never cools.
3. **Why this rig can:** STS3215 `Present_Temperature` is already read between rollouts (`scripts/run_s29_session.py`, 58 / 50 °C pause rule); no human resets, so heat accumulates.
4. **Experiment:** factor: hottest-servo temperature (thirds of the session); baseline: the coldest third; statistic: tracking gap (C2's metric) and success per third, logistic on temperature with episode order as a covariate; seed bar: hottest third tracking gap ≥ 1.5× coldest, CI excluding 1. n: 60 rollouts, one session, thermal pause disabled only below 58 °C (safety rule kept).
5. **Reuses:** SmolVLA bowls checkpoint, the session runner, C2's metric. New: per-step temperature logging (one bus read per chunk).
6. **Cost:** 1 arm day, 0.5 Orin day, $0, 0 teleop, ~5 owner h (at the rig, layouts reset by the harness).
7. **Figure:** tracking gap and rolling success vs hottest-servo temperature across the session.
8. **If null:** the 58 / 50 °C rule is enough; every session-length result on this arm is safe from heat (a paragraph for the register's methods).
9. **Objection:** "temperature and order are confounded with policy drift or lighting." Answer: order is a covariate, and a cool-down break at mid-session splits order from temperature.
10. **Venue:** ICRA 2027 workshop; short.
11. **Scores:** N4 · F4 · R2 · C4 · null 3 = **17**.

### C4 · Self-occlusion: where a latent world model's error goes when the arm hides the scene
1. **Question:** is W1's prediction error concentrated in frames where the arm occludes the target bowl in the top view, and is it the wrist view that carries those frames?
2. **Why new:** LeWM 2603.19312 (Push-T, full view), DINO-WM 2411.04983 (sim, full view), V-JEPA 2-AC 2506.09985 (Franka, unobstructed tabletop), Fast LeWM 2606.26217, SALT 2609.33595. Their benchmarks are rendered or staged so the workspace is visible; self-occlusion by the arm is absent by construction, so error vs occlusion was never stratified.
3. **Why this rig can:** one oblique top camera where the arm covers the bowl during transport in every episode; colour masks from the deterministic judge give the visible bowl fraction per frame for free.
4. **Experiment:** factor: visible fraction of the target bowl (G9c-3 masks), 4 bins; baseline: fully visible frames; statistic: W1 L2 error per bin, episode bootstrap; seed bar: error at < 25 % visible ≥ 1.5× fully visible; second arm: W1 (top + wrist encoder) with the wrist input zeroed at test, seed bar: occluded-bin error rises ≥ 30 % more than the visible bin's. n: 182 demos (val split) + 69 rollouts.
5. **Reuses:** W1 checkpoint (strict load), G9c-3 colour masks, the W1-A scoring script. New: per-frame mask fraction.
6. **Cost:** 0 arm, 1–2 Orin days, $0, 0 teleop, ~0.5 owner h.
7. **Figure:** W1 error vs visible bowl fraction, both views vs wrist zeroed.
8. **If null:** the latent ignores the bowl (error flat because nothing about the bowl is predicted) — which explains G1b's ranking failure and is itself the finding.
9. **Objection:** "occlusion co-occurs with fast transport motion." Answer: control for joint speed (regression with both); report the partial effect.
10. **Venue:** CoRL 2027 world-model workshop; short.
11. **Scores:** N3 · F5 · R5 · C3 · null 4 = **20**.

### C5 · Does the world model know the arm's limits?
1. **Question:** fed action sequences that command the SO-101 past its joint or reach limits, does W1 imagine the arm stopping at the limit (physics) or following the command (copying actions)?
2. **Why new:** LeWM 2603.19312, DINO-WM 2411.04983, V-JEPA 2-AC 2506.09985, DREAM-Chunk 2606.18589, Temporal-Distance JEPA 2607.25337. Their arms (simulated agents, Franka) reach every commanded pose in the workspace, and planners clip actions to the valid box, so out-of-limit imagination is never queried.
3. **Why this rig can:** the SO-101's short reach and the rig's joint limits and velocity clamp give recorded episodes where command and position diverge at the limit, the ground truth for the test.
4. **Experiment:** factor: commanded overshoot past the limit (0, 5, 10, 20°); baseline: the clamped trajectory actually recorded; statistic: joint-state probe (ridge, §30's) decoded from imagined latents vs the clamped and the commanded trajectories, error ratio; seed bar: decoded position closer to clamped than commanded for ≥ 80 % of overshoot steps at 10°. n: clamp events in the 69 + 182 episodes (count first; stop if < 30).
5. **Reuses:** W1, §30 ridge probes, clamp proxy events from smooth-chunking gate A. New: overshoot injection script.
6. **Cost:** 0 arm, 1 Orin day, $0, 0 teleop, ~0.5 owner h.
7. **Figure:** decoded joint angle under overshoot, imagined vs clamped vs commanded.
8. **If null (follows the command):** W1 copies actions rather than modelling the arm, consistent with §28 ("W1 reads its actions") — a sharper statement of why its planning fails.
9. **Objection:** "the §30 probe was not proven." Answer: §30 failed on the *beyond-state* question; here the z → joints probe must first pass a stated precondition on teleop val (seed R² ≥ 0.9), and the random-init W1 that blocked §30 is run alongside.
10. **Venue:** CoRL 2027 world-model workshop; short.
11. **Scores:** N4 · F4 · R4 · C3 · null 3 = **18**.

### C6 · What a teleop hour buys on a $200 arm
1. **Question:** what is SmolVLA's success per teleop hour on the bowls task at 45, 90 and 180 demos, and where does it flatten?
2. **Why new:** ACT 2304.13705 (50 demos, fixed), SO-101 benchmark 2606.08881 (fixed per-task sets), LoRA on SO-101 2512.11921 (200 demos), Mulligan 2610.05882 (budgets in episodes), Robot Data Factory 2609.16705. Demo counts are reported, operator time is not: staff or fleets collect, so the binding cost of a one-person rig is unmeasured.
3. **Why this rig can:** teleop is the binding constraint (~2 arm days a week), and the 182 bowls demos carry recording timestamps, so operator minutes per demo are known.
4. **Experiment:** factor: demos (45 / 90 / 180, nested subsets, both tasks); baseline: 180; statistic: success at 20 trials per arm, Wilson CIs, success per operator hour; seed bar: 90 within 10 points of 180 (flat) or not. n: 3 × 20 trials.
5. **Reuses:** `so101_bowls_all_trimmed`, the SmolVLA recipe, the session harness, G9c-3 judge. New: two fine-tunes (45, 90).
6. **Cost:** 1.5 arm days, ~$6 vast, 0 teleop, ~4 owner h.
7. **Figure:** success vs operator hours with CIs, the 50-demo literature point marked.
8. **If null (no flattening):** the rig is data-starved at 180 — the number every hobby SO-101 user needs.
9. **Objection:** "one task." Answer: two instructions; and the claim is the cost curve of this kind of rig, stated as such.
10. **Venue:** RA-L letter or CoRL 2027 workshop; short.
11. **Scores:** N3 · F3 · R4 · C5 · null 5 = **20**.

### C7 · How many own demos the community pool replaces
1. **Question:** does co-training on the 14.4 h of sensor-matched community SO-101 data let 45 own demos reach 180-demo success?
2. **Why new:** Ego2Robot 2608.02580 (synthesised human data), Xiaomi-Robotics-1 2607.15330 (100K h), MolmoAct2 2605.02881 (own released SO-100/101 data), SO-101 benchmark 2606.08881. Large-data papers own consistent fleets; the community pool is 71 small, heterogeneous sets (median 0.16 h) with different cameras and calibration — a regime only a hobby rig has to use.
3. **Why this rig can:** the catalogue and strict filter already exist (`data/community/REPORT.md`: 203 sets / 50 h; 71 strict / 14.4 h; 128×128 cache).
4. **Experiment:** factor: training mix (45 own; 45 own + pool; 180 own); baseline: 45 own; statistic: success, 20 trials each, Wilson; seed bar: 45 + pool ≥ 45 own + 15 points. n: 3 × 20 (shares C6's 45 and 180 arms).
5. **Reuses:** community cache, C6's checkpoints. New: one co-trained fine-tune.
6. **Cost:** 0.5 arm day beyond C6, ~$8 vast, 0 teleop, ~2 owner h.
7. **Figure:** success for the three mixes, the pool drawn as equivalent own demos.
8. **If null:** community data does not transfer at this scale — a warning for the SO-101 community, with the mismatch audit as the explanation.
9. **Objection:** "calibration differences, not data, explain it." Answer: the strict filter matches units and views; report a calibration-normalised arm as a secondary.
10. **Venue:** CoRL 2027 workshop; short.
11. **Scores:** N3 · F3 · R4 · C4 · null 4 = **18**.

### C8 · Counterfactual demos and language binding in a small VLA
1. **Question:** how many same-layout counterfactual demo pairs (same scene, other instruction) does SmolVLA need before its chunks depend on the instruction?
2. **Why new:** CoVer-VLA 2602.12281 (verifier over rephrasings), π0.7 2604.15483 (prompt steering), StreamPI 2608.26067, RoboHiMan 2510.13149. Multi-task corpora give many instructions per scene by volume; on one arm with two tasks, the layout predicts the task and language is optional (the rig's TurboVLA coin-flip and multitask README: 22 of 62 episodes had the other cube) — the shortcut only a small dataset exposes.
3. **Why this rig can:** the offline task-swap probe already exists (TurboVLA, multitask next step 1), and the multitask plan already lists same-layout pairs as the next data.
4. **Experiment:** factor: counterfactual pairs k ∈ {0, 10, 20} per colour; baseline: k = 0 (30k checkpoint); statistic: offline task-swap divergence of the first chunk endpoint (cm) on held-out two-cube frames, then success at choosing the named cube, 20 trials; seed bar: divergence > 3 cm on ≥ 80 % of frames at k = 20, and success ≥ +20 points.
5. **Reuses:** multitask SmolVLA 30k, the task-swap probe, 21 kept trials. New: 40 teleop episodes, two fine-tunes.
6. **Cost:** 1 arm day, ~$6 vast, 40 teleop episodes (~0.5 teleop day), ~4 owner h.
7. **Figure:** instruction sensitivity (task-swap divergence) and success vs k.
8. **If null:** language does not bind at this scale even with pairs — then frozen-vision fine-tuning is the bottleneck, which the multitask plan tests next.
9. **Objection:** "this is a data fix, not a finding." Answer: the dose–response k is the finding; no paper reports how many pairs bind language.
10. **Venue:** CoRL 2027 workshop; short.
11. **Scores:** N4 · F3 · R3 · C5 · null 4 = **19**.

### C9 · Memory forced by self-occlusion
1. **Question:** does giving SmolVLA two past top-view frames reduce failures that start while the arm hides the target bowl?
2. **Why new:** HyMeS 2608.09410 (memory as code; cues removed on purpose), Workspace Models 2609.20820, RoboTTT 2607.15275 (8K-step context), StreamPI 2608.26067. Memory tasks are staged by hiding a cue; none measures memory on occlusion the robot itself causes, because multi-camera rigs rarely lose the target.
3. **Why this rig can:** one top view that the arm occludes in every transport; G15 stages and failure onsets say when.
4. **Experiment:** factor: history (current frame vs current + frames at −0.5 s and −1 s); baseline: current frame; statistic: share of failures whose onset falls in an occluded window, and overall success, 20 trials each; seed bar: occlusion-onset failures halved and success not lower.
5. **Reuses:** bowls data, SmolVLA recipe, onset labels, C4's mask fraction. New: history input in the fine-tune and in vla.cpp.
6. **Cost:** 1 arm day, ~$3 vast, 0 teleop, ~3 owner h; vla.cpp change (~1 Orin day).
7. **Figure:** failures by occlusion state at onset, with and without history.
8. **If null:** occlusion does not drive failures here (they start in visible frames) — a negative that redirects memory work on this rig.
9. **Objection:** "20 trials cannot show halving." Answer: the primary is the onset share pooled over both arms' failures, with the trial count fixed in advance; a 40-trial version is the stated fallback.
10. **Venue:** ICRA 2027 workshop; short.
11. **Scores:** N4 · F3 · R3 · C4 · null 2 = **16**.

### C10 · The realised control rate as a hidden variable
1. **Question:** how far does the realised control rate of SO-101 rollouts drift from the nominal 30 Hz, and does running SmolVLA at 20 / 25 / 30 Hz change success?
2. **Why new:** RTC 2506.07339, AAC 2604.04161, ARLI 2608.23831 (inference delay as a training variable), Real-Time EXPO-FT 2609.18207. These assume a fixed control rate on real-time stacks and treat delay as the only timing variable; a USB servo bus and a Python loop make the rate itself vary — the rig's multitask rollouts ran at ~21.5 Hz while labelled 30 fps.
3. **Why this rig can:** every recorded episode has timestamps; the time-compression finding is already documented (multitask NOTES).
4. **Experiment:** (a) offline: realised-rate distribution over all own recordings (demos, rollouts), per session; (b) arm: factor rate (20 / 25 / 30 Hz, enforced), baseline 30 Hz, 20 trials each, Wilson; seed bar: success at 20 Hz within 10 points of 30 Hz (robust) or not.
5. **Reuses:** all recordings' timestamps, the session runner, SmolVLA bowls. New: a rate governor (sleep-to-deadline).
6. **Cost:** 1 arm day (b only), 0.5 Orin day, $0, 0 teleop, ~3 owner h.
7. **Figure:** realised-rate histogram per session, and success vs enforced rate.
8. **If null:** chunked policies are rate-robust in this range — a check other SO-101 users can skip.
9. **Objection:** "overlaps the edge-inference paper." Answer: that paper measures inference latency; this measures the control loop's rate with inference held fixed. Drop if the owner sees them as one.
10. **Venue:** ICRA 2027 workshop; short.
11. **Scores:** N3 · F4 · R4 · C3 · null 4 = **18**.

---

## Ranked

| rank | candidate | N | F | R | C | null | sum | arm days | vast $ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C1 Sub-servo smoothing | 4 | 5 | 5 | 4 | 4 | **22** | 0 | 0 |
| 2 | C2 Choosing or tracking | 4 | 5 | 5 | 4 | 3 | **21** | 0 | 0 |
| 3 | C4 Self-occlusion in the world model | 3 | 5 | 5 | 3 | 4 | **20** | 0 | 0 |
| 3 | C6 What a teleop hour buys | 3 | 3 | 4 | 5 | 5 | **20** | 1.5 | ~6 |
| 5 | C8 Counterfactual demos and language binding | 4 | 3 | 3 | 5 | 4 | **19** | 1 | ~6 |
| 6 | C5 Does the world model know the arm's limits? | 4 | 4 | 4 | 3 | 3 | **18** | 0 | 0 |
| 6 | C7 Community pool vs own demos | 3 | 3 | 4 | 4 | 4 | **18** | 0.5 (+C6) | ~8 |
| 6 | C10 Realised control rate | 3 | 4 | 4 | 3 | 4 | **18** | 1 | 0 |
| 9 | C3 Servo heat across a session | 4 | 4 | 2 | 4 | 3 | **17** | 1 | 0 |
| 10 | C9 Memory forced by self-occlusion | 4 | 3 | 3 | 4 | 2 | **16** | 1 | ~3 |

C1, C2 and C3 make one paper together ("execution on a $200 arm"); C6 and C7 share arms; C4 feeds C9's masks.

## Appendix — dropped from the ten

1. **The delay price of best-of-N** — at fixed wall-clock, does sequential test-time sampling on the Orin still help? Dropped: overlaps WS-081 A5 (DREAM-Chunk re-test), where the delay should be reported.
2. **Thermal latency drift at 25 W over an hour** — does p99 inference latency drift with SoC temperature? Dropped: inside the excluded edge-inference recipe.
3. **Judge cascade (deterministic check, then a VLM where it abstains)** — dropped: monitors / judges cluster overlaps the excluded failure-monitors subject; it is WS-081's B card (JEV).
4. **Operator-stopped vs timed-out episodes as a label source** — dropped: the length-baseline subject (excluded).
5. **Scripted policy as the baseline for every VLA post** — dropped: a re-test (COAP, WS-081 B).
6. **Gripper rim tolerance** — what bowl-rim offset makes a grasp fail, measured with the scripted policy. Dropped: weak reason (task-specific, no cluster gap).
7. **Teleop artefacts (pauses, re-grasps) in demos and their imitation** — dropped: overlaps C6 and needs hand labels of 182 demos (owner hours).
8. **Camera-pose drift between sessions** — does a few-mm top-camera shift between days explain day-to-day success variance? Dropped: needs repeated sessions (> 6 arm days for power).
9. **Contact sensing from servo current (NEXT on the SO-101)** — dropped: a re-test (FACTR 2, WS-081 B).
10. **Instruction vs layout conflict trials** — dropped: folds into C8's success arm (same layouts, both instructions).

## Novelty check for C1 / C2 (chat, 2026-10-31 night, alphaXiv search beyond the library)

Two papers measure or exploit the command–execution gap directly; neither is on an SO-101 and neither
decomposes a policy's failure into "chose wrong" vs "was not followed". Add both to Drive; C2's protocol cites
them as the closest prior work and positions against them.

- **2609.37334 — Taming VLAs under Robot Execution Errors: Self-Compensation and Stress Testing** (POSTECH /
  MBZUAI, 29 Sep 2026). Deployment-time online update of a VLA from the residual between the commanded action
  and the executed motion, no rewards or labels; RoboStress, a sim benchmark of friction / backlash / compliance /
  gravity-compensation error in seven scenarios; two physical AgileX Piper arms with different wear, +30 points
  average success each. **Closest to C2.** They *compensate* the gap and show it matters; they do not measure how
  much of a failed rollout is attributable to it, and the Piper is a 6-DoF industrial-grade arm, not a $200 servo
  arm where the gap is the norm. C2's attribution question survives; its method section must cite the residual
  signal as theirs. A cheap extra arm for C2: their self-compensation (residual-driven update) as the intervention,
  with the attribution as the pre-registered diagnostic — a possible register entry in its own right (B-list:
  needs an arm day).
- **2609.33145 — Beyond State-as-Action: Exploiting Command–State Discrepancy for Robot Imitation Learning**
  (Tsinghua / Imperial et al., 27 Sep 2026). Argues the discrepancy between commanded and measured state carries
  control information (contact, interaction constraints) and should not be erased by training on measured state as
  the action target. **Closest to C1**, and a caution for the whole rig: every LeRobot SO-101 dataset records
  `action` = leader-arm position and `observation.state` = follower position, so the discrepancy is in the data
  already; the register's "error vs recorded continuation" metric uses the leader command. C1 must state which
  side its jerk and clamp numbers are measured on.

No paper found that reports servo-tracking numbers for the SO-101 / STS3215 as a factor in policy success; the
closest public material is in LeRobot issues and the SO-101 benchmark paper (2606.08881) already in the register,
which does not separate the two sources. The structural reason in C2's card stands.
