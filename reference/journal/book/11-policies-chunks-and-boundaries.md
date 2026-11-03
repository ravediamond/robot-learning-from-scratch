# Chapter 11. Policies, chunks and boundaries

## 11.1 Behaviour cloning from first principles

A policy is a function π: observation → action. Behaviour cloning (BC) learns it by supervised regression on demonstration pairs (o_t, a_t): minimise a loss between π(o_t) and the demonstrated a_t. For continuous actions the loss is L1 or L2 on the action vector (ACT uses L1; so does TurboVLA), or a likelihood under a generative head (diffusion, flow matching: SmolVLA, VLA-JEPA, π0).

What BC assumes and where it breaks on your data:

- **The demonstration is the target.** Teleop demos include operator noise, hesitation, and corrections. BC learns all of it. The "tidy-loop" demos with deliberate corrections are valuable precisely because they show recovery, which clean demos never do.
- **The state at test time looks like the state in the data.** It does not, after the first small error: the policy drifts into states the demos never visited (covariate shift). Chunking, corrections in the data, and DAgger-style relabelling all exist to mitigate this. Your 24 % oscillating failures (G14) are covariate shift made visible: the policy enters a pose between two demonstrated behaviours and alternates.
- **The loss rewards what is predictable.** A target that the state already predicts (the next 0.4 s of a smooth reach) gives the model no reason to read the instruction or the image (fact 7). This is the deepest point in the chapter: BC does not learn what you want; it learns what reduces loss, and on teleop data the cheapest reduction is usually "continue the current motion".

**Multimodality.** When the demos contain two valid behaviours from the same state (go left or go right), an L1/L2 head regresses to their mean, which is neither. ACT addresses this with a CVAE latent at training time (zero at inference). Diffusion and flow-matching heads represent the distribution and sample one mode. TurboVLA's plain L1 head "hovered between the bowls" (fact 3): mode averaging on a bimodal target. Whether a generative head would fix it is hypothesis H2, untested.

## 11.2 Action chunking

Predict H future actions a_{t..t+H−1} from o_t; execute some or all before re-querying. ACT introduced it with H = 100 at 50 Hz; SmolVLA uses 50; TurboVLA 12.

Three distinct reasons chunks help, usually conflated:

1. **Temporal consistency.** One decision for H steps avoids H independent noisy decisions.
2. **Horizon reduction.** Fewer decisions per episode means fewer opportunities to compound error (covariate shift grows with decision count).
3. **Latency cover.** The arm has H actions to execute while the next inference runs; on the Orin, 238 ms of inference is ~7 ticks, so a chunk shorter than ~10 would starve the controller.

And one reason they hurt: the world changes during the chunk. A chunk planned at t is executed open-loop until the next query; if an object moves or a grasp slips at t + 10, the remaining 40 actions are wrong. The paper in the prep queue (2610.01626) proposes a *stability statistic* that measures how much the environment changes over a chunk and predicts when long chunks fail. On a static tabletop with bowls it should say "long is fine"; on the LEGO cell with bricks that shift, it should not.

**Temporal ensembling** (ACT): query every step, keep the overlapping predictions for each future timestep, execute their exponentially weighted average. Smooths boundaries at the cost of a query per step; the TurboVLA plugin implements it; too expensive on the Orin for SmolVLA.

## 11.3 Flow matching for action heads

SmolVLA's action expert, VLA-JEPA's DiT and π0 all generate the chunk by flow matching. From first principles:

Define a path from noise to data: x_t = t·ε + (1 − t)·a, with ε ~ N(0, I), a the demonstrated chunk, t ∈ [0, 1]. (SmolVLA and vla.cpp use this convention: t = 1 is pure noise, t = 0 is the action.) The velocity along the path is dx_t/dt = ε − a. Train a network v_θ(x_t, t | o) to predict ε − a from a noised chunk and the observation (plus the VLM tokens), with an MSE loss. At inference: sample ε, set x = ε, and integrate dx = −v_θ(x, t | o)·dt from t = 1 to 0 in N Euler steps (N = 10 by default). The result is a sample from the learned action distribution.

What this means in practice:

- **Each inference is stochastic.** Two calls on the same observation give different chunks (the H0 spread of 1.08° was the *within-observation* sample spread for VLA-JEPA; PyTorch SmolVLA's noise-to-noise spread was 12.2° max |Δ| on the parity frames). Parity between implementations requires *sharing ε* (chapter 6).
- **N controls latency.** Ten steps = ten expert forward passes per chunk; most of the 238 ms. Fewer steps with the plain Euler schedule degrade; the "fewer steps" paper (2609.21216) changes the schedule to keep quality. VLA-Rolling instead starts the next chunk's integration from the previous chunk's unexecuted tail (warm start), cutting steps for the same quality.
- **The VLM is frozen in SmolVLA fine-tunes;** only the expert and projections train. So "SmolVLA learned the task" means the expert learned to map frozen vision-language tokens to actions. The P² paper's note that plain-VLM features are weaker probes than action-expert features is consistent with this.

## 11.4 Real-time execution and the boundary problem

Synchronous loop: query, wait 238 ms (arm idle or repeating the last action), execute, repeat. Blocked 14 % of the time at 26.6 Hz in your measurement; visible as a pause at every chunk.

Asynchronous loop: query in the background while executing the current chunk; when the new chunk arrives, skip the actions that have already elapsed (inference delay d ≈ 7 steps) and continue from the new chunk's step d. The problem: the new chunk was sampled from fresh noise, so its step-d action does not match where the arm actually is at that moment. The joint command jumps. On your data the median jump at d = 8/12/20 was 1 028 ms-equivalent "naive" (the G13b rows report it as the jump statistic) and the arm visibly jerks.

Two fixes compared in G13b on the same 254 boundaries:

- **Real-Time Chunking (RTC, 2506.07339)**: during the new chunk's denoising, add a guidance term that pulls the first d actions toward the previous chunk's tail (soft-masked inpainting). Exact RTC needs gradients through the expert at each Euler step; in vla.cpp that means backward rules ggml lacks for flash attention and concat, so it is not available there; in PyTorch it cost ~2× per chunk and cut the jump 21–41 %.
- **Prefix overwrite** (gradient-free): after denoising, replace the first d actions of the new chunk with the previous chunk's corresponding tail, then let the chunk continue. No gradients, +0.4 ms in vla.cpp, 72–74 % jump reduction, drift unchanged, prefix error 0.0 by construction, default path bit-identical. The weaker-than-expected exact-RTC number ("weaker than LeRobot's identity mode, to check") is a flag in the register that nobody has resolved; a reviewer will ask.

Why prefix overwrite can win: the boundary error is dominated by the *first few* actions disagreeing with the arm's state, and copying the tail fixes exactly those; RTC's guidance is a soft constraint that trades off against the policy's own prior. The live test (G13a) is whether the 72 % offline reduction becomes smoother motion and no worse success on the arm.

## 11.5 Oscillation and register tokens

G14: 24 % of failed episodes (20.3 % of all 69, the figure comparable with REKI's 18.6 %) show pose oscillation (reversals at ≥ 3 consecutive chunks, cos < −0.5, > 3° per chunk), real re-grasp loops and pan dithering on inspection. Mechanism (11.1): a state between two demonstrated behaviours, alternating. Two proposed fixes: SMILE's post-hoc chunk smoothing/outlier filter (cheap, prep queue 2.4), and REKI's register tokens, a small KV-cache memory that gives the policy a persistent context across chunks so it does not re-decide from scratch each time (§13, ~$10, needs the port). The prep-queue bar for SMILE is written so that a pass removes §13 from the plan: the cheap fix first.

## 11.6 The sensitivity gate

Before any policy goes to the arm, five minutes offline (`sensitivity_gate_v3.py`):

- **Start reach**: from the rest pose observation, how far does the first chunk move? (< 10° means it learned to stay; the idle-start bug.)
- **Task-swap**: same frame, change the instruction; how much does the chunk change? (TurboVLA 0.1–0.25°; SmolVLA 7–13°; the LIBERO-Goal control on a correct TurboVLA 0.105 vs a 0.095 line, ~5.5°.)
- **Per-layout**: does the "left bowl" instruction steer further left than "right bowl" across layouts? (chance = 50 %; SmolVLA 86 %.)
- **Plan-to-plan jump** (informational): consecutive chunk disagreement.

A policy that fails any of the first three has not learned to condition on what the task requires, however good its loss. The gate generalises to any VLA; it is the single most reusable artefact of the TurboVLA work and belongs upstream.

## 11.7 Which policy, when

- **ACT**: fastest (59 ms), no language, one checkpoint per task, strong on single-task bowls (10/10). The baseline for "does a learned policy work on this rig at all".
- **SmolVLA 450M**: language-conditioned, 21 tasks in one checkpoint, 238 ms via vla.cpp, the workhorse. Frozen VLM: its language grounding is whatever the pretrained VLM gives.
- **VLA-JEPA 2B**: same score as SmolVLA on bowls at 3× training cost; its world-model loss is training-time only; the inference-time world model (reranking) was a no-go. Kept for the harder-primitive comparison.
- **TurboVLA**: the port diverged; faithful reproduction pending. Its value, if it works, is no LLM at inference (CPU-capable).

The decision rule in the register: compare on the hardest primitive, not on bowls.

## 11.8 Exercises

1. Write the flow-matching forward path, loss and Euler sampler in 40 lines of PyTorch for a 2-D toy action distribution with two modes. Show that L1 regression gives the mean and flow matching samples a mode.
2. From the G13b data, reproduce the "naive" boundary jump at d = 8 for five boundaries by hand: previous chunk's action at step d vs new chunk's action at step d.
3. Run the sensitivity gate on the 21-task SmolVLA and on ACT. Explain ACT's task-swap number.
4. Write one paragraph on why a 12-step chunk is more vulnerable to fact 7 than a 50-step chunk, in terms of what fraction of training targets depend on the instruction.
