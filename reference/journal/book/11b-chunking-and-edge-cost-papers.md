# Chapter 11b. Chunk length, candidate selection and denoising depth: three inference-time papers, and the tick budget

Chapter 11 explains chunks, boundaries and the prefix overwrite; 11a reads RTC, the paper behind it. This chapter reads the three inference-time papers tested between 2026-10-27 and 2026-10-30, and two read in the same week whose tests were decided on paper (VeriVLA, not tested; FlashRT-PAI, bar frozen). All three change only what happens at inference — no retraining — which makes them the cheapest family to test on this rig, and all three were tested with $0 of compute on the Orin. They share one lesson, and it is the reason the chapter exists: **an inference-time method is a claim about a latency budget, and the budget is the hardware's, not the paper's.** Two of the three are right about the mechanism on SmolVLA; none transfers as published to a Jetson Orin NX.

| # | Paper | arXiv | One line | Verdict here |
|---|---|---|---|---|
| 1 | Adaptive Action Chunking (AAC; Liang et al., CVPR 2026) | 2604.04161 | Chunk length chosen per decision from the entropy of N sampled chunks | **fail** (A1, A3, A4) |
| 2 | PACT-VLA (Zhu et al.; HKUST-GZ / One Robotics) | 2610.28019 | Generate N candidate chunks, let the frozen policy pick one by its own velocity residual under the newest frame | **fail** as published; the selection itself replicates offline |
| 3 | Horizon × depth (Yang et al.; UCAS / CASIA / NUS / SJTU) | 2610.32102 | Fewer Euler steps pay off under long open-loop commitment, because truncation contracts the action cloud | **not proven** |
| 3a | VeriVLA (HKU) | 2610.21985 | Trigger on sample disagreement, then pick among 10 chunks with a V-JEPA 2 progress model | not tested (decided by H0, G1b and the Orin budget) |
| 3b | FlashRT-PAI (AnyDevice AI & Alps Lab) | 2610.24349 | Fusion and launch overlap: 1.3–3.7× on eager VLA pipelines | bar frozen, not run |

Shared setup for all three, so it is stated once: policy `smolvla_2026-09-24_14-34-40` @ `c1d76f6` (H = 50, 10 Euler steps); the 20 G13b episodes of `so101_bowls_all` (254 decision points per delay); the Orin NX, PyTorch at 25 W for selection and distribution measurements, vla.cpp at MAXN_SUPER + `jetson_clocks` for every latency number (the 238 ms reference chunk). Every protocol was frozen and committed before its first number: AAC `581a955` / `12ecf1a`, PACT-VLA and Horizon × depth `0a79846` / `15bcb89` / `f010075`.

---

## 11b.1 Adaptive Action Chunking (Liang et al., CVPR 2026; 2604.04161)

### Method

At a decision frame, draw N = 20 chunks from the same observation with independent noise. For each future step i, fit a covariance Σ_i over the N samples of the continuous action and take the Gaussian entropy E_i = ½ log[(2πe)^k det Σ_i]; for the gripper, the binary entropy of the closed fraction. The running mean Ē_h = (1/h) Σ_{i<h} E_i is the uncertainty of committing to h steps. The executed length is h* = max(argmax_h (Ē_{h+1} − Ē_h), ξ): commit until the point where uncertainty starts rising fastest, never less than a floor ξ. The paper's intuition: long chunks in free-space transport, short ones near contact.

### What was translated, and the deviations

- The arm acts in absolute joint angles; the paper's floor ξ is defined on translation / quaternion magnitudes, which do not exist here. ξ = 2 steps, fixed in the prompt before the run.
- The paper does not say which of the N samples is executed. Sample 0 was chosen, because it shares its noise with the fixed-length arm and makes the comparison paired.
- Gate 2 (on the arm) was never planned: the bowls task sits at 9/10 for SmolVLA, so a 15-point gain cannot be shown. That is chapter 2's ceiling rule applied before the run, not after.

### Bars and numbers

| Bar (frozen) | Value | Verdict |
|---|---|---|
| A1: median h* in transport ≥ median h* in the grasp window + 2 | transport **2** (n = 72), grasp **2** (n = 24) | fail |
| A3: AAC + prefix overwrite ≤ fixed + overwrite on boundary jump, d = 8 | **0.973** vs 0.940 | fail |
| A4: a 20-sample decision ≤ 298 ms in vla.cpp (25 % over 238 ms) | N = 1: 241 ms, N = 20: **4,751 ms**, N = 40: 9,506 ms | fail |

A2 (descriptive): AAC's earlier switch alone lowers the naive jump from 3.42 to 2.28, because the new chunk is sampled closer in time; with the prefix overwrite both arms are already at about one normal step (0.94 vs 0.97).

### The mechanism, found post-hoc

h* equals the floor in 219 of 254 decisions (86 %). SmolVLA's sample spread grows fastest at the very first step (median entropy 1.11 → 2.68 → 3.32 nats over steps 1–3) and keeps growing; the steepest rise of Ē_h is at h = 1 in 75 % of decisions, so the rule returns ξ. On this policy the entropy rule does not track task phases; it degenerates to "near-minimal chunks". This reading is post-hoc and labelled so in the register; it explains A1, it does not change the verdict.

### What is not there

- No timing of the N-sample decision in the paper's robot setting, and no batched-sampling requirement stated. vla.cpp has no batched sampler, so the cost is N sequential chunks; a batched path would have to make 20 samples cost 1.25 chunks to meet A4.
- No test of whether the entropy profile of a small flow-matching expert has the phase structure the rule needs. That is the premise that failed here.

### For the SO-101

Two independent reasons say no: the rule does not do what it is for on this policy (A1), and it costs twenty chunk inferences per decision on the Orin (A4). Either alone decides it. The useful residue is the post-hoc measurement: SmolVLA's uncertainty is front-loaded, which is a fact about the policy worth knowing before any uncertainty-driven method (adaptive chunking, sample-and-select, uncertainty-triggered replanning) is tried on it.

---

## 11b.2 PACT-VLA (Zhu et al.; 2610.28019)

### Method

Under asynchronous inference, generate N = 4 candidate chunks; one control step before the switch, take the newest observation and score each candidate by **FlowRes**, the residual of the frozen policy's own velocity field on that candidate under that observation; commit the candidate with the lowest residual. Training-free, no verifier, and the paper states **no added control-step delay**. Reported: real bimanual PiperX with π0.5, 60 % (naive async) → 82 %; Kinetix at a 4-step delay, +15.0 points over RTC. Their compute: RTX 5090 / H100 / RTX Pro 6000, 35–49 ms per chunk, d = 2 at 30 Hz.

### The translation

Here a chunk costs 238 ms, d = 8 steps: four times their delay and twice their largest tested one. 151 ms of the chunk is the vision encoder, and the GPU is 94–98 % busy during generation, so the extra observation pass cannot hide behind it. The paper's own fixed-pool test (its App. D.2) already showed that FlowRes under the stale frame equals a random pick: the value is the fresher frame. So the protocol split the claim in two, which is the whole design lesson of this section:

- **Gate B, selection quality, offline.** Does PACT pick chunks closer to the recorded continuation than a random pick (B-P1) and than FlowRes under the stale frame (B-P2)?
- **Gate A, the cost.** Is d_PACT ≤ d_naive on the Orin, i.e. is "no added delay" true here?
- B-P3, transfer: PACT paying its measured delay still beats naive at d = 8.

Cheap baselines run beside it: naive async (one sample), the exact random-pick expectation, and **mean-of-4**, which the paper never ran.

### Measurement checks before the numbers

Candidate 0 bit-identical to a fresh single sample; the recorded future scores below the median candidate on 68 % of boundaries (bar ≥ 60 %); FlowRes with a shuffled observation does not pass B-P1 (−0.04° vs random). One implementation fix was committed before any number: `predict` returns 6-dim chunks, padded to the model's action dimension for the velocity query, the residual read on the 6 real dimensions only.

### Numbers

Mean |Δ| to the recorded actions over each selector's executed window, 5 arm joints, 20 episodes:

| Selector | d = 8 | d = 12 |
|---|---|---|
| single (naive async) | 2.47° | 2.66° |
| random of 4 | 2.54° | 2.76° |
| FlowRes, stale frame | 2.54° | 2.72° |
| **PACT** | **2.27°** | **2.47°** |
| mean of 4 | 2.32° | 2.51° |
| oracle | 1.79° | 1.89° |

B-P1 PACT − random −0.27° (episode bootstrap 95 % CI −0.44 to −0.09): **pass**. B-P2 PACT − stale −0.27° (−0.48 to −0.09): **pass**. Stale FlowRes = random (+0.00°), exactly as the paper's own appendix predicts.

Gate A, the cycle on the Orin: T_gen(4) sequential 949.9 ms + vision and prefill on the pre-commit frame 165.7 ms + verification 84.5 ms (projected: vla.cpp has no velocity query on a given x_τ, so it is the PyTorch ratio 16.0 × vla.cpp's 5.3 ms per denoise step) = **1,200 ms** (1,034.5 ms if everything overlaps perfectly), against 237.5 ms for one chunk. In 33.3 ms ticks: **d_PACT 37 (optimistic 32) vs d_naive 8**. Gate A **fails**.

B-P3 is **not computable**: with d = 37, a 50-step chunk has 13 steps left, fewer than the 25 the protocol's executed window needs. No other window was chosen after seeing this.

### Verdict and why the register says "fail"

The experiment's own summary is "replicates offline, does not transfer to the Orin". The register's paper verdict is **fail**, because the claim that fails is one the paper states — no added delay — and it fails by 29 ticks. The claims build of 2026-10-30 lists it as one of five contradictions between a paper and this rig (C-020-2 vs C-020-3, "conditions that differ: none"). Read chapter 2.6a for how that row is computed.

### What is not there

- Any latency table where N candidates are generated without a batched sampler, or on a device where vision dominates the chunk.
- Mean-of-N. Here it sits between single and PACT (2.32° vs 2.27°); a selector that needs no velocity query should be in every sample-and-select paper's table.

### For the SO-101

The selection works: −0.27° is real, the CI excludes zero, and the measurement checks passed. What does not exist is the time to do it. The decision rule written before the run said a gate-A fail means a register entry and no executor change. The general rule, now in chapter 6 and in the interview chapter: **time the method's cycle on the target device before measuring its quality**; quality measured offline is a statement about an infinitely fast computer.

---

## 11b.3 Horizon × depth (Yang et al.; 2610.32102)

### Claim and mechanism

With frozen flow-matching VLAs, cutting Euler steps from 10 to 2 hurts under frequent replanning and helps under long open-loop commitment. The stated mechanism (their Proposition 1): truncating the integration contracts the cloud of sampled actions around an almost unchanged centroid, so a long committed chunk is less likely to wander. π0 on LIBERO-Long: (s, N) = (5, 10) 81.2 % → (20, 2) 90.5 %, interaction +6.4 ± 2.1; on a Franka FR3, 2,400 trials, a sign reversal on four quasi-static tasks. A side claim: N = 2 avoids thermal throttling on edge devices.

### The translation

This rig already runs s = H = 50, synchronous: the paper's long-commitment regime. So the paper's large latency gain (5.6×, mostly from moving s 5 → 20) is not available; only the step cut is. The cheap baseline is doing nothing: N = 10 at s = 50. The paper's own grid at s = 50 gives N = 2 75 % vs N = 10 66 % for π0, so the comparison is exactly its full-commitment row.

### Bars and numbers

| Bar (frozen) | Value | Verdict |
|---|---|---|
| A1: median R(2)/R(10) ≤ 0.70, centroid shift δμ ≤ 0.05, max centroid |Δ| ≤ 3.0°; fail if R > 0.85 | R(2)/R(10) **0.78** (CI 0.73–0.81); δμ 0.018; 2.35° | not proven |
| A2: N = 2 ≤ 0.85 × N = 10 per chunk in vla.cpp, and N = 10 within ±5 % of 238 ms | **0.83** (196.1 vs 237.5 ms); N = 10 −0.2 % | pass |

A1 measures 20 frames × 32 draws × N ∈ {1, 2, 3, 5, 10}. The contraction is real and the centroid does not move, as Proposition 1 says, but it is 22 % here against the paper's 49 %; the CI lies between the bar and the fail line. Check: at N = 10, draw 0 equals the normal path (max |Δ| 0.0°).

A2's latency split is the instructive part: per chunk at N = 10, vision 150.7 ms, prefill 14.9 ms, **5.3 ms per denoise step**. Cutting 8 of 10 steps saves 41 ms (17 %), not 80 %. On the Orin the denoiser is not where the time goes.

A3 (descriptive): N = 2 sits slightly closer to the demonstration at every execution horizon (s = 50: 1.53° vs 1.67°, 142 boundaries).

A4 (thermal, with a reading rule written before the run): no throttling at N = 10 in either power mode (MAXN: clock 1,173 MHz throughout, 68.6 °C max; 25 W: 63.2 °C max). The thermal claim has nothing to act on at this cadence. Two instrument notes worth keeping: this JetPack's `tegrastats` prints no GPU clock, so the first MAXN run could not be read and the script was changed to read devfreq before re-running; and at 25 W the governor holds the clock at its floor, so a "drop below 95 % of the first-minute median" rule cannot see throttling there. Both are in the README; the second is labelled post-hoc.

### Decision

The arm gate (B: 20 + 20 trials, N = 2 non-inferior within one success) was conditional on A1 and A2 both passing. A1 is not proven, so B was not run and SmolVLA stays at 10 steps. The verdict is **not proven**, not fail: the mechanism is present at half size, and the result that would decide (success on the arm) was never measured. Chapter 5 explains why that word is the right one.

### For the SO-101

A 17 % cheaper chunk is available for the price of a 20-trial arm session, if A1 is re-asked as a question the arm can answer. The general lesson is the A2 split: before any method that speeds up the action head, measure what fraction of the chunk the action head is. Here it is 53 of 237 ms.

---

## 11b.3a VeriVLA (University of Hong Kong; 2610.21985) — read, not tested

### Claim

Keep a flow-matching VLA frozen. A training-free **trigger** fires when two sampled chunks disagree (cross-sample variance) or vary a lot in time (temporal variation), each normalised by mean action magnitude. When it fires, sample 10 chunks and pick the one with the highest predicted progress from a V-JEPA 2 latent world model with a task-conditioned progress head (labels from Robometer; predictor pretrained on 23,000 DROID trajectories). PiperX dual arm, 50 demonstrations per task, 50 trials: completion score 50.8 % → 65.5 %; +326 ms per triggered decision on an RTX 4090. The trigger alone: Status AUC 0.727 for contact-critical moments, in simulation.

### Why not tested

Every part of the selection half has already been measured here and failed: best-of-N reranking with a latent world model (H0); latent ranking against the commanded joints (G1b, regret 0.263, and a perfect top-2 swap still floors at 0.133, so no better scorer rescues the proposal pool); the V-JEPA 2 encoder next to SmolVLA on the Orin (the V-JEPA 2.1 / RoboJEPA latency result); Robometer as a progress labeller on SO-101 data (fails zero-shot); and ten SmolVLA samples at ~238 ms each (this chapter). The paper does not report the commanded-joint or copy-last comparator that decided G1b. That is chapter 1's rule: a pass would change nothing that is not already decided.

### What it changes

The trigger is new and cheap: the policy's own sample disagreement, with no world model. It is added as a fourth runtime monitor to NEXT_ANGLES §2, scored on the standing REAP bar beside the logistic risk head, with its temporal-variation axis alone as the commanded-joint baseline. It needs two chunks per call from fixed seeds during the next rollout session; with 4 failures among the 30 SmolVLA rollouts on hand it cannot be scored now. The session must log whether the second sample delayed the executed chunk — on the Orin it costs one more chunk if done serially, the same budget problem as PACT-VLA.

## 11b.3b FlashRT-PAI (AnyDevice AI & Alps Lab; 2610.24349) — bar frozen, not run

### Claim

In eager π0.5 on a Jetson AGX Thor, non-matrix kernels between the GEMMs take 31.49 % of decision latency and launch gaps 22.52 %; kernel fusion plus launch overlap make policies 1.3–3.7× faster than the official compiled paths at matched 16-bit precision.

### Translation

Their pipelines are PyTorch eager or `torch.compile`: thousands of Python-launched kernels. vla.cpp builds one ggml graph per call. So the paper's diagnosis may not apply at all, and the cheapest evidence is already in hand: `tegrastats` reads 94–98 % GPU load during timed calls, which caps idle time near 2–6 %. The paper never compares against a single-graph engine.

### Bar (frozen, $0, Orin only)

From an `nsys` trace of 20 timed calls: gap fraction = 1 − busy / wall; "other" share = non-GEMM, non-attention, non-copy kernel time / busy. Headroom exists if other ≥ 25 % or gap ≥ 15 % (→ a vla.cpp fusion entry and an upstream issue). No headroom if other < 15 % and gap < 5 % (→ latency work stops at kernel level and goes to fewer denoising steps). Between: not proven. The bootstrap is over calls of one process, and the draft says so.

### Why it belongs here

It is the measurement that 11b.3's A2 split asks for next: vision is 151 of 238 ms, and whether that 151 ms is compute or overhead decides whether any kernel work can move the tick budget that sank PACT-VLA and AAC.

## 11b.4 How this family evaluates, side by side

| Paper | Their device | Their per-chunk cost | Here | Latency in the paper's tables? | Cheapest baseline in the paper? |
|---|---|---|---|---|---|
| AAC | not reported for the N-sample decision | — | 241 ms × N sequential | no | fixed chunk length, yes |
| PACT-VLA | RTX 5090 / H100 / RTX Pro 6000 | 35–49 ms, d = 2 | 1,200 ms cycle, d = 37 | yes, on their GPUs | random pick, yes; **mean-of-N, no** |
| Horizon × depth | A100 (sim), Franka FR3 | not decomposed | 237.5 ms, of which 151 ms vision | per-step totals, not split | N = 10 at each s, yes |

Read with chapters 3, 6 and 10:

- **All three are statements about a latency budget.** The paper's budget is a datacentre GPU or a workstation; this rig's is an Orin where one chunk already costs 8 ticks. A method that multiplies chunks (AAC ×20, PACT ×4) multiplies the dominant cost; a method that shortens the denoiser saves only the denoiser's share.
- **The first measurement for any inference-time method is its cycle on the target device.** It is cheap (an hour, $0), it can decide the experiment alone (AAC A4, PACT A), and none of the three papers reports it in a form that transfers.
- **Split "does it select well" from "can it run".** PACT's offline pass is a real result that would have been lost if the experiment had gone straight to the arm, where the latency would have masked it as a plain fail.
- **The cheap baseline the paper skipped is often close.** Mean-of-4 lands 0.05° from PACT with no velocity query.

## 11b.5 Corrections and additions to chapter 11 and the register

- 11.4 (boundaries): add AAC's A2 row as a second independent case where the prefix overwrite already brings the jump to about one normal step; any boundary method must now beat overwrite, not naive async.
- 11.7 (which policy, when): add the chunk-cost split for vla.cpp on the Orin (vision 150.7 / prefill 14.9 / 5.3 ms per denoise step at MAXN) as the standing input to every latency argument.
- Chapter 10 (hardware): the power mode is part of every latency number. Timings at 25 W PyTorch and at MAXN_SUPER + `jetson_clocks` in vla.cpp are different instruments; a protocol names one.
- Chapter 6: add "time the cycle on the device before reading quality" as a standing check for inference-time methods; add the devfreq read for the GPU clock.
- Register: PACT-VLA's offline selection result is kept as its own row (B-P1, B-P2 pass) beside the paper verdict, so the replicating part is not lost under "fail".

## 11b.6 Exercises

1. From the A2 split, write the latency of a SmolVLA chunk as a function of N Euler steps and of a hypothetical batched sampler that costs 1 + 0.1 (k − 1) chunks for k samples. At which k does PACT meet d_PACT ≤ d_naive? At which does AAC meet A4?
2. Re-derive AAC's h* by hand for one decision from `results/posthoc_hstar.json`: list E_1 … E_5, Ē_1 … Ē_5 and the differences; show why the argmax is h = 1.
3. PACT B-P3 was declared not computable rather than re-windowed. Write the alternative window you would have been tempted to use, and the sentence that explains why choosing it after seeing d = 37 would invalidate the result.
4. Add mean-of-N to the PACT table at N = 2, 4, 8 offline. Decide, before running, what difference from PACT would make you drop the velocity query entirely.
5. Write the 20 + 20 arm protocol for Horizon × depth's gate B as if A1 had passed: layouts, labelling at rest blind to the arm, the non-inferiority margin, and the ceiling rule if N = 10 scores ≥ 18/20.
