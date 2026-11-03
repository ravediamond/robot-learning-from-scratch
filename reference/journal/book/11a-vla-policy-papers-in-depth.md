# Chapter 11a. The VLA policy papers, in depth

Chapter 11 explains what a chunked policy is and what your boundary and TurboVLA gates found. This chapter is the reading behind it: eight papers, each read from the method and experiment sections, with the objective, the training setup, every evaluation protocol with metric and n, the headline numbers with table references, what was not run, and what each means on a 6-DoF position-controlled arm with two webcams and a 1024-core GPU.

| # | Paper | arXiv | One line |
|---|---|---|---|
| 1 | SmolVLA | 2506.01844 | Your policy. 450M, flow-matching expert, half the VLM; the ablation tables are the manual |
| 2 | TurboVLA | 2607.27205 | V+L→A without an LLM; 0.2B, 31 ms; the paper your port diverged from |
| 3 | VLA-JEPA | 2602.10098 | Latent-world-model pretraining on human video for a VLA; what it does and does not buy |
| 4 | REKI | 2610.22178 | Register memory in the action expert; oscillation 18.6 → 4.1 % |
| 5 | Real-Time Chunking (RTC) | 2506.07339 | Inpainting across chunk boundaries; the method your prefix overwrite approximates |
| 6 | VLAs-as-Tools | 2605.13119 | VLA as a bounded callable tool under a VLM planner; progress feedback instead of polling |
| 7 | Augmentation in VLA post-training | 2610.05994 | Augment the critic, never the actor; +7.8 OOD |
| 8 | SO-101 VLA benchmark | 2606.08881 | π0.5 / SmolVLA / Wall-X / ACT on your arm; failure taxonomy; recovery rate |

---

## 11a.1 SmolVLA (Shukor, Aubakirova, Capuano et al., Hugging Face; 2506.01844)

### Method

Two parts. (i) **VLM**: SmolVLM-2 (SigLIP vision encoder → SmolLM2 decoder), global image only at 512×512, **no tiling**, pixel shuffle to **64 visual tokens per frame**; state projected to one token; visual + language + state tokens through the decoder, but **only the first N = L/2 = 16 LLM layers** are run — the action expert reads features from layer N, the rest of the VLM is discarded. (ii) **Action expert** v_θ: a transformer with **interleaved cross-attention (to the VLM's keys/values) and causal self-attention (within the chunk)**, hidden size 0.75 × d, trained by conditional flow matching:

  L^τ(θ) = E_{p(A_t|o_t), q(A_t^τ|A_t)} ‖ v_θ(A_t^τ, o_t) − u(A_t^τ | A_t) ‖²,  A_t^τ = τ A_t + (1−τ) ε,  ε ~ N(0, I),  u = ε − A_t

τ from a Beta distribution; 10 Euler steps at inference. Only the action expert is trained; the VLM is frozen. Total 450M, ≈100M in the expert. Chunk n = 50.

Pretraining data: 481 community datasets, 22.9k episodes, 10.6M frames — all SO-100 — filtered by embodiment, episode count, quality and frame coverage; task strings rewritten by Qwen2.5-VL-3B from sampled frames; cameras manually mapped to OBS_IMAGE_1/2/3 (top, wrist, side) because community naming is inconsistent and "a consistent camera ordering is rather beneficial". 200k steps, global batch 256, AdamW, cosine 1e-4 → 2.5e-6, bf16, torch.compile, 4 GPUs, ≈30k GPU-hours in total for the project.

Asynchronous inference (Algorithm 1): the client consumes a queue and triggers a new chunk when |A_t|/n < g; near-duplicate observations (joint-space distance < ε) are dropped; the new chunk is aggregated with the old on the overlap. g = 0 is synchronous (idle E[ℓ_S] per chunk), g = 1 is a forward pass per tick; the analytic condition for never emptying is g ≥ E[ℓ_S]/(Δt · n).

### Evaluation protocol

- Simulation: LIBERO (4 suites × 10 tasks, **10 trials per task**, binary) and Meta-World (50 tasks, 10 trials each). Multi-task training, 100k steps batch 64.
- Real: SO-100 pick-place, stacking, sorting; SO-101 pick-place-LEGO; **50 demos per task = 10 per each of 5 starting positions**; fine-grained scoring (0.5 grasp + 0.5 place; sorting 0.25 × 4); multi-task fine-tune 200k steps; synchronous inference for real-world numbers. Trial counts for the real tables are not stated in the text; the figure-5 speed test uses 10 trials × 5 positions.
- Baselines: π0 (3.3B, fine-tuned on the target data) and ACT (trained from scratch per task).

### Numbers

- Table 2 LIBERO avg: SmolVLA 0.24B 82.75, 0.45B 87.3, 2.25B 88.75; π0 3.3B (robot-pretrained) 86.0; OpenVLA 76.5; Diffusion Policy 72.4. Meta-World 0.45B 60.0 vs π0 47.9.
- Table 3 SO-100 real (avg): ACT 48.3, π0 61.7, SmolVLA 78.3. Table 4 SO-101 LEGO: ACT 70 / 40 (ID / OOD positions), SmolVLA 90 / 50.
- Table 5: pretraining matters more than anything else in the paper — multi-task without community pretraining 51.7, with 78.3; single-task no pretraining 40.
- Fig. 5 async vs sync: same success (73.3 vs 78.3), task time 9.7 s vs 13.75 s, **19 vs 9 cubes in fixed time**.
- Ablations, LIBERO avg (all from scratch, frozen VLM): CA+SA 85.5 vs CA 79.0 vs SA 74.5 (Table 6); causal 74.5 vs bidirectional 67.5 (Table 7); N = 16 layers 78.5, N = 32 80.3, skip-every-second 75.5, VLM-256M 75.8 (Table 8); expert width ×0.75 77.5, ×1.0 82.3, ×0.5 80.3 (Table 9, non-monotone); flow matching 80.25 vs L1 regression 75.25 (Table 10, the long-horizon suite 53 vs 38 is where it comes from); state into the VLM as prefix 80.3 vs into the expert 73.3 (Table 11); chunk 1 / 10 / 30 / 50 / 100 → 50.0 / 84.0 / 78.5 / 80.3 / 74.5 (Table 12); **actions executed before re-observing 1 / 10 / 30 / 50 → 80.3 / 82.8 / 70.8 / 51.8 (Table 13)**.

### What is not there

- Seeds or intervals anywhere; 10 trials per LIBERO task is the convention and the Wilson interval on 90 % at n = 10 is 60–98 %.
- A no-language ablation (does the instruction do anything on single-task real data?). TurboVLA's Table 3 is the only paper here that runs one; your fact 7 is the SO-101 version.
- The async success rates are *lower* than sync (73.3 vs 78.3) and the paper reports it; the claim is throughput, not accuracy.

### For the SO-101

Three things to carry into every SmolVLA run. (1) **Table 13 is your boundary problem in the authors' own numbers**: executing the full chunk before re-observing costs 29 points on LIBERO; your 238 ms chunk at 30 Hz is ~7 steps of delay, which the table says is nearly free, but only if you re-observe — which is what the prefix overwrite and RTC are for. (2) **Table 11**: state belongs in the VLM prefix, not the expert; check the plugin you run does this. (3) Table 5 says the 22.9k-episode community pretraining is worth +27 points on SO-100; any fine-tune you start from scratch is competing against that and should say so. The async stack is what vla.cpp replaces: measure your g equivalent (chunk refill threshold) and state it in the protocol.

---

## 11a.2 TurboVLA (Xie, Yao, Wu et al., HUST / Huawei; 2607.27205)

### Method

Replace V → L → A (vision projected into an LLM, actions decoded from LLM features) with **V + L → A**: DINOv3 ViT-B (or ViT-L) patch features + BERT token features, projected to d = 256, fused by **N = 6 bidirectional cross-attention layers initialised from Grounding DINO's feature-enhancement weights**, concatenated, and decoded with an **ACT-style transformer decoder** with H = 12 learnable action queries and the robot state as separate tokens into the decoder memory only ("the robot state is required for translating task-conditioned scene features into executable actions but not necessary for visual-language correspondence"). Loss: plain ℓ1 on the chunk, behaviour cloning, no auxiliary objective, no language-model loss. 0.2B total, 0.9 GB VRAM, 31.2 ms per chunk on an RTX 4090.

Equations: Z^l = P_l(f_text(x)) (full token sequence kept, eq. 4); Z^{v,(i)} = P_v(f_img(I^{(i)})) + E_pos + e_view, views concatenated (eq. 5); Z^s = f_state(s) (eq. 6); (V^ℓ, L^ℓ) = FusionLayer_ℓ(V^{ℓ−1}, L^{ℓ−1}) for ℓ = 1…N (eq. 7); Z^{vl} = [V^N; L^N] (eq. 8); Â = D_θ(Q_a, [Z^{vl}; Z^s]) (eq. 9).

### Evaluation protocol

- LIBERO: one mixed-suite model, ViT-B, 12-step chunks of 7-DoF actions, **80k steps, 10k warm-up, effective batch 256**, lr 5e-5, four RTX 4090; VLA-Adapter rollout protocol, **50 rollouts per task, 2,000 trials total**.
- RoboTwin 2.0 (bimanual, 50 tasks, clean setting only): ViT-L, 50-step chunks of 14-D absolute joints, 55k steps, batch 192, StarVLA framework, 100 rollouts per task.
- Real: AgileX Piper, wrist + third-view D435; four tasks; LIBERO checkpoint fine-tuned on 4 × 65 teleop demos for 12.5k steps; **40 trials per task**; π0.5 under the same data and protocol.
- Efficiency: latency on RTX 4090 batch 1, from multimodal input to a chunk; VRAM = peak of the full online policy; baselines re-measured with official code and checkpoints.

### Numbers

- Table 1 LIBERO avg: TurboVLA 97.7 (99.2 / 99.8 / 97.4 / 94.2); π0.5 96.9 at 93.6 ms and 3.4B; VLA-JEPA 97.2 at 108.7 ms and 2.8B; SmolVLA 88.8 at 203.1 ms; OpenVLA-OFT 97.1 at 112.2 ms; Diffusion Policy 72.4 at 924.8 ms.
- Table 2 RoboTwin multi-task: 60.2 at 43.4 ms vs π0.5 57.0 at 95.6 ms, StarVLA-α 50.3; the per-task ACT row is 29.7.
- Fig. 4 real: 92.5 / 80 / 90 / 87.5 vs π0.5 ≈ 87 / 76 / 82 / 85 (read from the bar chart).
- Ablations (§5.4): **remove language 97.7 → 70.8, LIBERO-Goal 97.4 → 11.6** (Table 3); a learned task-ID embedding instead of text recovers to 2.3 below full; T5-small 97.1, SigLIP text 95.5 (Table 4); direct concatenation 95.2, one-way cross-attention 96.1 / 96.5, bidirectional 97.7 (Table 5); N = 2 → 6 → 8: 93.5 → 97.7 → 96.6 (Table 6); H = 8 / 12 / 15: 96.4 / 97.7 / 95.6 (Fig. 6).

### What is not there

- No intervals; LIBERO at 50 rollouts per task is better than most, the real robot at 40 per task has no CI and no seeds.
- No ablation of the Grounding DINO initialisation — the one you most need, since your port had none. The paper states the init once (§5.1) and never tests it.
- No language-removal ablation on the *real* tasks; the 70.8 is LIBERO, where suites share scenes and the instruction is the only disambiguator (LIBERO-Goal: same layout, different goals, hence the 11.6).
- No comparison at matched chunk length to the ACT baseline (ACT with the same DINOv3 features would isolate the fusion module).

### For the SO-101

This is the reference for chapter 8 and §15 of NEXT_ANGLES; now with the paper open, the four port divergences map to specific sentences: fusion init from Grounding DINO (§5.1, never ablated — so path A is also the first ablation of it anyone has run); bidirectional cross-attention with shared affinity (eq. 7, Table 5: the one-way variants cost 1.2–1.6 points on LIBERO, which is within what your gate can resolve only with n ≥ 100); state into the decoder memory, not the fusion (eq. 6/9 — your port prepended it to vision before fusion, which lets the fusion attend to state and is a different model); ACT-style decoder with MLP head. The most important number for your fact 7 is Table 3's LIBERO-Goal 11.6: **on data where the instruction is the only thing that distinguishes targets, this architecture uses it**. Your data had the instruction redundant with state on 94 % of frames; the paper's result does not contradict yours — it says the architecture can read language when the data forces it to, and your data did not. That sentence is the TurboVLA correction row.

**Path A result (2026-10-23).** The faithful model — upstream trainer, Grounding DINO fusion init via the full LIBERO checkpoint (666/672 tensors), `BiAttentionBlock`, state into the decoder memory, 35k steps, batch 64 — moves its actions by **0.020°** on a task swap over the bowls data (frames 0.006–0.053°), against the released checkpoint's 0.105 on LIBERO-Goal. The four port divergences, removed, changed nothing on the quantity that mattered. Fact 7 is the full story; the architecture claim is withdrawn; and the Grounding DINO initialisation, never ablated in the paper, is ablated here on one dataset with no effect on instruction use. The paper and the result agree once the data is named: V+L→A reads language when the instruction decides, and copies the state when it does not.

---

## 11a.3 VLA-JEPA (Sun, Zhang, Qi et al., USTC; 2602.10098)

### Method

A VLA (Qwen3-VL backbone, SigLIP-2 vision, flow-matching action head) pretrained on **human video without actions** through a latent world model, so that learned "latent action" tokens carry state-transition structure instead of pixel-change or leaked future.

- World-state encoder: frozen V-JEPA 2 per view, views concatenated: s_{t_i} = ‖_v F(I_{v,t_i}) (eq. 1).
- Latent action tokens ⟨latent_i⟩ (each replicated K times) are read out of the VLM given the **initial frame only** plus instruction: z_{t_i} = p^VLM(⟨latent_i⟩ | {I_{j,t_0}}, ℓ) (eq. 2). No future frame ever enters the VLM — the leakage-free principle.
- Autoregressive latent world model predicts the next state chunk from past states and z: ŝ_{t_{i:i+1}} = p^WM(s_{t_{0:i}}, z_{t_{0:i}}) (eq. 3); time-causal attention across steps, full within a step.
- L_WM = Σ_k ‖ŝ_{t_k} − s_{t_k}‖ (eq. 5), teacher-forced; read as the ELBO of eq. 4 with the KL term vanishing for a deterministic frozen target.
- Robot data: ⟨action⟩ tokens appended after the latent tokens give z_a (eq. 6); flow matching L_FM = E ‖v_θ(a_t, t | z_a) − (a_{0:H} − ε)‖² (eq. 8); joint L = L_FM + β L_WM (eq. 9).

Pretraining: Something-Something-v2 (220k human videos) with eq. 5, DROID (76k demos) with eq. 9, all VLM parameters trained; fine-tune on LIBERO (2k demos), Fractal / BridgeV2 for SimplerEnv, 100 real demos across three tasks; 8 A100.

### Evaluation protocol and numbers

- LIBERO (Table 1): **50 episodes per task, 500 per suite**; VLA-JEPA 97.2 avg (96.2 / 99.6 / 97.2 / 95.8), π0.5 96.9, OpenVLA-OFT 97.1, UniVLA 95.2, LAPA 65.7; **without human video 96.1** (−1.1).
- SimplerEnv (Table 2): Google Robot avg 65.2 vs π0-Fast 67.5 (best), WidowX 57.3 tie with LAPA; **without human video Google Robot 78.4** — *better* than with, which the authors attribute to real-to-sim gap and the value of in-distribution expert data.
- LIBERO-Plus (Table 3, seven perturbation axes): 79.5 avg vs OpenVLA-OFT 69.6, π0 53.6; Language 85.4, Light 95.6, Background 93.6, Layout 85.1; Camera 63.3 (π0-Fast 65.1 better); **without human video 62.9** — the −16.6 is where human video pays.
- Real Franka (Fig. 4, 100 demos, three tasks): ID 0.70 vs π0 0.57 vs π0.5 0.37; OOD task 0.17 vs π0.5 0.20; OOD layout 0.47 vs 0.37. Trial counts in Appendix B.
- Fig. 5: success vs proportion of human video rises on every perturbation axis; Table 4: video horizon T = 8 best on average, T = 4 on Spatial, T = 16 on Goal.

### What is not there

- No intervals; the ±1 differences in Table 1 are not resolvable at 500 episodes per suite (binomial SE ≈ 0.7 at 97 %).
- No ablation of the leakage-free design against a leaky variant (future frames into the VLM); the argument is made in §1 by reference to LAPA's attention maps (Fig. 6), not by a controlled run.
- The real-robot comparison fine-tunes π0 / π0.5 on the same 100 demos, which is right; n per task is in an appendix.

### For the SO-101

The honest reading is: **pretraining on human video buys robustness to perturbation (LIBERO-Plus +16.6), not in-distribution accuracy (LIBERO +1.1, SimplerEnv −13)**. Your VLA-JEPA at 177 ms on the Orin (vla.cpp benchmark) is the model this paper describes. If you ever run a perturbation gate (lighting, §8) with two policies, this paper predicts which one survives and why; it does not predict a success difference on nominal cube trials, and a nominal gate that showed none would confirm it rather than refute it. The method lesson is the pair of ablation rows "w/o human videos" placed under every table — a one-line control that turns a model paper into an evidence paper.

---

## 11a.4 REKI (Marrie, Pacaud, Ramanana Rahary, Chen, Böhle; Kyutai / Inria; 2610.22178)

### Method

A 2.4B VLA (Qwen3.5-2B VLM + 10-block, width-768 flow-matching expert) pretrained only on public data, plus **register memory**: n_r = 32 learnable register tokens per frame are prepended to the noisy action chunk in the expert; they cross-attend the frame's encoded context, self-attend the registers of the K − 1 preceding frames (K = 3), never read the action tokens, skip the adaptive normalisation, so they depend only on the observations and are **cached as keys/values across chunks** (eq. 5). The VLM still encodes one frame at a time; a training window of K frames is one batched pass with a loss on every frame (eq. 6); eight (τ, ε) draws per frame share the encoder pass (multi-draw variance reduction). Inference: 21.8 ms per chunk on one H100 (prefill 9.5, Euler 8.1, cached registers +0.1).

Objective: ℓ_fm = E ‖v_θ(A^τ, τ | c) − (ε − A)‖² (eq. 2), averaged over the K frames of a window with each frame's chunk conditioned on contexts up to it (eq. 6). Conditioning: VLM final-layer hidden states read by cross-attention; state pooled into one embedding that modulates through AdaLN (zero-initialised); state dropped with p = 0.2; embodiment / frequency / episode-length prompt fields dropped 0.15–0.2.

Data: RoboSet, Galaxea, AgiBot World 2026, RoboCOIN, RoboMIND, RDT-1B, DROID, BridgeData V2, twelve OXE datasets, InternData-A1 and GR00T-X sims, bimanual YAM; three egocentric human datasets (~350k episodes) in Qwen-VLA's 32-D bimanual layout; vision-language data at a small ratio. Per-dataset quantile normalisation to [−1, 1], native control conventions kept, loss mask over zero-padded action width. 200k steps, 32 H100, global batch 2,880.

Augmentation (§5.3): colour jitter, mild crop, **random camera dropout, shuffled camera order**; mirroring (flip images, reflect state/actions, swap arms, prompt marks mirrored) on RoboTwin / RoboCasa; disabling camera dropout or image augmentation costs > 20 points on LIBERO-Plus camera-viewpoint changes; disabling state dropout costs 8.1 under initial-state changes (App. D.2).

### Evaluation protocol and numbers

- Benchmarks: LIBERO and LIBERO-Plus (7 perturbation axes); RoboTwin 2.0 in-distribution and **Clean2Rand** (finetune on clean, evaluate under six randomisation conditions, mean); RoboCasa365 (atomic / composite seen / composite unseen); RoboCasa-GR1. Rollouts per task in App. Table 3; baseline numbers taken from published results (App. Table 7) — so not re-run.
- Fig. 4: LIBERO-Plus 90.4 (Qwen-RobotManip 4.7B 89.0, π0.5 85.7, π0 69.4); Clean2Rand 65.8 (ABot-M0 55.7; Qwen-RobotManip 68.5 with in-house pretraining); RoboTwin ID 85.3 vs 94.1 best; RoboCasa365 38.4 (RLDX-1 36.0); LIBERO 97.5. No-pretraining REKI: 86.1 LIBERO-Plus, 52.0 Clean2Rand.
- Fig. 5a memory window (finetune K → eval K): 3→3 90.4 / 3→2 90.1 / 3→1 88.9 / 1→1 88.9 on LIBERO-Plus; RoboTwin 73.4 / 73.6 / 67.2 / 67.9; "most of the gain comes with the first past frame", and evaluating a K = 3 model at K = 1 ≈ a K = 1 model — the gain is from *reading* past frames at inference.
- Fig. 5b oscillation: episodes in which an arm reverses its motion at three or more consecutive chunks — **4.1 % with memory, 14.8 % when the same model is evaluated without, 18.6 % trained without**; two-arm tasks 24.5 → 3.1 (Fig. 5c). RoboTwin Clean, 50 tasks, **20 matched seeds each**. Mechanism offered: perceptual aliasing — two poses A and B each look like demo states continuing toward the other.

### What is not there

- Intervals on Fig. 4 are per-subset ticks, not CIs; baselines are quoted, not re-run under the same evaluation.
- No real-robot table in the main text ("generalises to real robots", details elsewhere).
- No ablation of n_r (32 registers) or of registers vs the simpler "feed the previous frame to the VLM" at matched compute, except by the cost argument in Fig. 2.

### For the SO-101

Your G14 found 24 % of episodes oscillating at chunk boundaries with a memoryless policy. REKI measured the same phenomenon with a defined criterion (direction reversal at ≥ 3 consecutive chunks), 20 matched seeds per task, and showed one past frame removes most of it. That is the cleanest external replication of a mechanism you have; cite it in the G14 post and adopt their criterion so the numbers are comparable. The cheap test on your hardware: SmolVLA has no memory, and adding registers means retraining; but the **evaluation half** of Fig. 5a — the same checkpoint run with and without the past-frame cache — is free for any policy that has one, and the oscillation criterion is a 30-line script over your existing rollouts. Also: the augmentation sentence (camera dropout and shuffled order are worth > 20 points on viewpoint shift) is the single most actionable training detail in this chapter for a two-webcam rig whose mounts move.

---

## 11a.5 Real-Time Chunking (Black, Galliker, Levine; Physical Intelligence; 2506.07339)

### The problem, in their notation

Chunk A_t = [a_t … a_{t+H−1}] from a flow policy integrated by A^{τ+1/n} = A^τ + (1/n) v_π(A^τ, o_t, τ) (eq. 1). Δt the control period, δ the inference time, **d = ⌊δ/Δt⌋ the inference delay in steps**; execution horizon s; asynchronous inference must start at s − d so a chunk is always available, but the junction between a_{s−1|0} and a_{s|s−d} is arbitrary — the new chunk, generated from a fresh noise draw, may have chosen a different strategy (Fig. 2's bifurcation). Temporal ensembling averages across chunks and "is not guaranteed to produce valid actions" when the distribution is multimodal.

### Method

Treat the next chunk as **inpainting**: the first d actions are frozen (they will be executed before the new chunk arrives), the H − s remaining overlapping actions are attended to with an exponentially decaying weight, the last s are free. Guidance is ΠGDM (eq. 2): add to the velocity field min(β, (1−τ)/(τ r_τ²)) · (Y − Â¹)ᵀ diag(W) ∂Â¹/∂A^τ, where Â¹ = A^τ + (1−τ) v (the one-step denoised estimate, eq. 3), r_τ² = (1−τ)²/(τ² + (1−τ)²) (eq. 4), Y the previous chunk right-padded, and **W** the soft mask (eq. 5): 1 for i < d, c_i (e^{c_i} − 1)/(e − 1) with c_i = (H − s − i)/(H − s − d + 1) for d ≤ i < H − s, 0 beyond. The guidance clip β is their addition; without it, the few-step samplers used in control were unstable (App. A.2). The gradient is a vector-Jacobian product through the policy — one backward pass per Euler step. Algorithm 1: a controller-side GetAction every Δt and a background InferenceLoop that estimates d conservatively as the max of a buffer of past delays.

### Evaluation protocol

- Simulation: **12 dynamic Kinetix tasks** (force control, throwing, catching, balancing, Gaussian action noise), chosen because quasi-static benchmarks are saturated by open-loop chunking; expert RPO policies × 6 seeds → 1M transitions; flow policy H = 8, MLP-Mixer; **2,048 rollouts per data point, 95 % Wilson intervals**; delay d ∈ 0…4; baselines naive async, BID (rejection sampling, 64 chunks), TE.
- Real: π0.5, bimanual 6-DoF, **H = 50, Δt = 20 ms, n = 5 denoising steps**; model latency 76 ms (baselines) / 97 ms (RTC), remote inference over LAN 10–20 ms → d ≈ 6; **+100 / +200 ms injected latency → d ≈ 11 / 16**. Six tasks (light candle, plug ethernet, make bed, shirt folding, batch folding, dishes in sink), each scored by substeps with a cutoff; **10 trials per task per method per delay = 480 episodes, 28 h of robot time**; metric **average throughput = task progress / episode duration**, ±1 SEM. Baselines: synchronous (s = 25 then pause), TE sparse (s = 25, async), TE dense (s = d).

### Numbers

- Fig. 5: RTC above every baseline at every delay; the gap to BID widens with delay; hard masking (freeze only d steps) is below soft masking at small d; RTC and BID improve monotonically as s decreases (closed-loop corrections pay only if continuity holds); TE is worst at every delay including d = 0.
- Fig. 6 right: throughput at +0 / +100 / +200 ms — RTC flat (≈ 0.8 tasks/min), synchronous falls linearly (≈ 0.7 → 0.5), **both TE variants trip the robot's protective stop at +100 ms and score 0**. "RTC completes tasks faster than synchronous even when inference pauses are removed" (Fig. 6 top, controller-steps axis) — fewer mistakes and retries, not just no pauses. Light candle (the only non-retryable task) shows the largest final-score gap.

### What is not there

- Compute: one backward pass per Euler step, so RTC is 97 vs 76 ms on their stack; on the Orin a backward through SmolVLA's expert in ggml does not exist (chapter 14.1.2).
- Position-controlled robots: the simulated benchmark is force-controlled, where asynchrony is necessary; they say synchronous is "a reasonable default" for position control and show it loses on throughput anyway.
- A boundary-jump metric on the real robot (they show one joint's trace in Fig. 1); the quantitative claim is throughput.

### For the SO-101

Your prefix overwrite is RTC with β → ∞ on the frozen d steps and W = 0 elsewhere — the "hard masking" line in Fig. 5, which the paper shows works but is below soft masking when d is small. That is a precise, published prediction for G-boundary: at your d ≈ 7 of H = 50, soft masking should cut the jump further than the −72 % you measured, at the cost of a VJP per Euler step you cannot run in vla.cpp today. The honest protocol order: (1) report the −72 % as the hard-mask point on their curve; (2) implement soft masking in PyTorch on the Orin (974 ms, off-line) and measure the jump on the same 254 boundaries — that is a gradient-free comparison of the *output*, no engine work; (3) only if (2) wins by more than the pairing noise, write the ggml backward rules. And adopt their metric: **throughput** (progress per minute) is what a smoother boundary is for; a jump statistic is the instrument check, not the result.

---

## 11a.6 VLAs-as-Tools (Lei, Liu, Xiong et al., SJTU; 2605.13119)

### Method

A high-level VLM agent (Gemini-3-Flash class, > 300B) plans; a family of VLA **tools** executes bounded invocations c_k = (g_k, z_k): a tool-family label (grasp / open / place …) and a scene-grounded instruction (eqs. 1–3). Each call runs for a bounded horizon H_k and returns a **progress signal p_t ∈ [0, 1]** predicted by an auxiliary head on the backbone feature — local to the call, not to the task — which a threshold monitor uses to advance, replan or continue, so the agent is not polled every step.

Tool-Aligned Post-Training (TAPT): (i) **invocation-aligned units** — demonstrations segmented into bounded windows (o, a, z, g, p\*), with p\* a progress target and g assigned by operational rules (contact pattern, force direction, object motion; App. A); RL rollouts start from states where the invocation would begin and are rewarded 1 if a local predicate ψ_{z,g}(s) (contact, containment, relative pose…) becomes true within H; (ii) **tool-family residual adapters** ΔW_g = B_g A_g (eq. 4), one low-rank bank per family, selected deterministically by g so the label changes the execution path rather than being another token; (iii) L_TAPT = E[−Σ log π_{θ,φ_g}(a_t | o_t, z, h_t) + (λ_prog/T) Σ ‖p̂_t − p\*_t‖²] (eq. 6); RL with GRPO on the same units.

### Evaluation protocol and numbers

- Backbones: OpenVLA, OpenVLA-OFT, π0.5. Benchmarks: LIBERO-Long, RoboTwin (8 single-arm tasks), CALVIN_D (80/20 split, native subtask structure as a check that gains are not an artefact of their own split), LIBERO-CF-Long (their own counterfactual reconstruction of unreleased LIBERO-CF: hand-edited goals on familiar layouts).
- Table 1 (IL): standalone SFT → direct planner → TAPT + interface: OpenVLA 77.2 → 77.0 → 82.4; OpenVLA-OFT 92.0 → 80.2 → 95.6; π0.5 92.4 → 80.4 → 97.2; RoboTwin π0.5 39.4 → 30.0 → 62.5. **Wrapping a standard SFT policy in a planner loses 0.2–12 points**; the gain comes from making the executor tool-aligned.
- Fig. 3 (RL): LIBERO-Long π0.5 standard RL 80.0 → invocation-aligned RL 89.0 → TAPT 89.2 → + interface 91.2; CALVIN 76.0 → 78.8 → 82.3.
- Table 2 (invocation fidelity, LIBERO-CF-Long): Faithful rate (completes the counterfactual goal) and Non-biased rate (does not drift to the original task): OpenVLA-OFT 19.4 / 31.2 → 54.0 / 47.4 with full TAPT; π0.5 24.8 / 39.6 → 54.8 / 54.6.
- Table 3 (component ablation, LIBERO-Long): interface alone on a non-tool-aligned executor 80.2 / 80.4 — "planner wrapping alone cannot make a non-tool-aligned executor reliable"; IA post-train *hurts* π0.5 (93.4 → 91.2) and the residual adapters recover it (96.0).
- Table 4: progress-triggered intervention vs VLM-every-5-steps: calls per task 0.228–1.988 vs 57–110; replan success 34–61 %.
- Table 5: adapters +9 % params, +7 % inference on OpenVLA-OFT.

### What is not there

- Real robot: none ("extending … to diverse real-robot settings is an important direction").
- Trial counts per cell are in an appendix (App. D); no intervals.
- LIBERO-CF-Long is the authors' own reconstruction; the original is unreleased.

### For the SO-101

This is the v1.4/v1.5 harness with numbers you do not have yet, and the numbers say two things your spec already assumed and one it did not. Assumed: the agent must stay out of the step loop (Table 4: 0.2–2 calls per task against 57–110); the executor returns a result the agent reads at rest. Not assumed: **a policy dropped behind a planner gets worse** (Table 1, −0.2 to −12) unless its training units are the invocation units. Your SmolVLA fine-tunes are whole-episode "put the blue bowl in the pink bowl"; under the harness they will be called as "grasp blue bowl" then "place in pink bowl", which is a distribution shift the paper measured. The honest line for G24 (policy replaces scripted executor): the first thing to try is segmenting the 69 rollouts into the harness's unit actions with your existing key-press labels as p\*, and fine-tuning on those units — TAPT's component (i) alone, which Table 3 says is worth ~1 point on LIBERO and much more on RoboTwin (39.4 → 52.4 for OFT's IA SFT row). The progress head is the P² idea from chapter 13.2 trained with a target; the harness's tier-2 verdict at rest can supply p\* = 1 at the end of a satisfied unit.

---

## 11a.7 How (and how not) to use augmentation in VLA post-training (Grooten, Vanschoren; TU/e; 2610.05994)

> Removed from the project KB on 2026-10-24 to make room for LaLa-World (12b.8a); this section is now the only copy of its numbers.

### Method

PPO post-training of π0.5 and GR00T N1.5 from a LIBERO-Spatial SFT checkpoint, 300 epochs × 512 episodes, RLinf; Flow-SDE to get action log-probabilities out of a flow head; a value head on the VLM's final features; VLM backbone **frozen** (only the action expert and value head train). Augmentations act only in the training forward pass — rollouts and evaluations always see clean images: **random overlay** ō = (1−α) o + α m with m from a 19,167-image ImageNet bank, fresh draw per sample, same overlay across both cameras; **random shift** 10 px pad-and-crop at 224 (4.5 %, matched to DrQ's 4 px at 84). Ablated: where (actor / critic / both), what (overlay / shift / both), how strong (α 0.1–0.9).

### Evaluation protocol

In-distribution: LIBERO-Spatial, 10 tasks × 50 seeds = 500 episodes per checkpoint. OOD: LIBERO-Plus, the five visual axes (textures 258, light 292, layout 385, camera 376, noise 351 = 1,662 tasks, each once), unweighted mean of per-axis success. **Evaluation noise floor measured directly**: one checkpoint through the full suite twice → 0.06 points (π0.5), 1.77 (GR00T); gaps below that are called indistinguishable. One training seed per configuration, stated as a limitation and defended by the breadth of the sweep.

### Numbers

- Table 1 (where): π0.5 SFT 56.7 OOD; PPO no aug 73.9; **actor+critic 0.1, actor-only 4.3 (collapsed, stopped at epoch 100); critic-only 81.7 (+7.8), IND 93.2 → 98.0**. GR00T: no aug 66.0; actor variants 0.0–0.1; critic-only overlay 66.0 (no gain with overlay).
- Table 2 (what, critic-only): π0.5 overlay 81.7, shift 78.9, both 75.9; GR00T overlay 66.0, **shift 76.0 (+10.0)**, both 69.0. Overlay helps most on sensor noise (+27.1 π0.5), shift on camera (+12.0 / +19.4) and layout (+14.8 GR00T). Composition is worse than the better single type.
- Table 3 (how strong, π0.5 overlay): α 0.1 → 0.9: 77.0 / 77.0 / 81.7 / 80.6 / 76.6; every α beats no augmentation.
- Mechanism (§5): with a frozen backbone the actor never sees an augmented pixel; the gain reaches it only through the advantage estimate Ã_t computed from a critic trained on perturbed views — so "our numbers are a lower bound" and the trainable-backbone version is the experiment they would run next.

### What is not there

- Real hardware; one benchmark; one RL algorithm; one seed per configuration (all stated).
- No IL-only control: whether critic-only augmentation beats simply augmenting the SFT stage is not asked.

### For the SO-101

You do not run PPO on the Orin and should not start. What transfers is the method: (1) **measure the evaluation noise floor by evaluating the same checkpoint twice** before claiming any gap — one afternoon on your 69 rollouts with a fixed-seed replay, and it belongs in chapter 6 as an instrument check; (2) "match the augmentation to the model's weakest axis" is a rule for your lighting angle (§8): run the perturbation suite first, find the axis where SmolVLA is worst, and augment that in the IL fine-tune — which is the IL-only control the paper did not run and the first thing a reviewer would ask them; (3) the collapse-when-the-actor-is-augmented result is specific to PPO clipping and should not be read as "never augment the policy" for IL — REKI augments the policy in IL and wins > 20 points on viewpoint.

---

## 11a.8 Benchmarking VLAs on SO-101: failure and recovery (Yu, Qiu; Hiroshima; 2606.08881)

### What it is

The only paper in the knowledge base on your arm. Four tasks chosen for the execution property they probe rather than semantic difficulty: Pen Transfer (control fidelity), Selective Color Sorting (grounding under distractors: "put the pink block in the plate and remove the others"), Multi-Object Packing (temporal consistency), Precision Pen Placement (insertion under embodiment noise). **100 teleop demos per task, poses randomised within a workspace; 400 demos total; four policies (π0.5, SmolVLA, Wall-X, ACT) fine-tuned with their native objectives and minimal interface changes; 20 trials per model-task pair = 320 episodes.** Datasets released.

Metrics: success rate (final object state, no intervention); a four-way failure taxonomy per failed trial (grasp instability, repetition loop, state mismatch, precision misalignment), one *primary* mode per failure; semantic vs execution decomposition (state mismatch = semantic; mean of grasp instability and repetition loop = execution; precision misalignment excluded); **recovery rate = successful recoveries / recovery opportunities**, where a recovery is "restores a valid task state after an execution failure and continues without intervention".

### Numbers

- Table 3 success: ACT 75 / 0 / 10 / 50 = 33.75; SmolVLA 70 / 5 / 10 / 45 = 32.5; Wall-X 95 / 0 / 30 / 80 = 51.25; π0.5 95 / 10 / 55 / 65 = 56.25. **Color Sorting ≤ 10 % for everyone**; Pen Transfer 70–95 for everyone.
- Table 4 failure-mode distribution over failed trials: grasp instability 91–100 % and repetition loop 91–100 % for all four (modes are not exclusive in the table); state mismatch ACT 98.1, SmolVLA 70.4, Wall-X 61.5, **π0.5 45.7**; precision misalignment Wall-X 0.
- Table 5 recovery rate: ACT 6.45, SmolVLA 3.23, Wall-X 20.51, π0.5 30.77.
- The authors' own caveat on SmolVLA vs ACT (32.5 vs 33.75): "the difference corresponds to only a small number of successful trials and may be influenced by the limited evaluation budget".

### What is not there

- Intervals: at n = 20 the Wilson interval on 10 % is 3–30 % and on 55 % is 34–74 %; most between-model differences in Table 3 are not resolvable except Pen Transfer vs Color Sorting. The paper does not compute them.
- Labelling: the taxonomy is assigned by the authors; no inter-rater agreement, no recording of who labelled.
- A length or policy-identity baseline for the failure labels; no camera or lighting description; no latency per policy on the Orin or whatever ran them.
- Any explanation of why grasp instability and repetition loop both exceed 90 % when "each failed trial is assigned a single primary failure mode".

### For the SO-101

Three uses. (1) It is the external prior for your success rates: a language-grounding task with distractors is ≤ 10 % for every current VLA on this arm with 100 demos — so your bowl-identity failure mode (wrong object, fact 7) is the field's, not your port's. (2) The **recovery-rate** metric is one you can compute from your existing labels (a failure event followed by a valid state and continued progress), and it separates policies that success rate cannot (π0.5 30.8 vs SmolVLA 3.2 at similar success on three of four tasks). Add it to the register's per-episode record. (3) The taxonomy is a starting point for the G9c-3 failure types, but their Table 4 shows what happens without exclusivity and an end rule: every column near 100 %. Yours has an end rule (stillness or timeout) and one verdict per episode; keep it. If you cite this paper's numbers, cite them with the Wilson intervals you compute, since the authors did not.

---

## 11a.8a System 2 on a $200 arm: FineART (2609.36416) and G0.5 — bars frozen 2026-10-30, running

### What is claimed

FineART (Choghari et al., Scale AI × Hugging Face; arXiv preprint, venue unconfirmed): a dataset of 40,543 episodes with 533,913 subtask labels and a subtask-mid-trained VLA; subtask supervision raised a spatial-disambiguation task from 32 % to 100 %. The LeRobot "Prompting Robots" post (Part 2) ships `lerobot-annotate` (VLM subtask and box labels), an interactive rollout runtime (`/subtask`, `/autosteer`, `/vqa`) and a G0.5 case study on an SO-101 whose own conclusion is the interesting one: on a simple task, subtask and box prediction did not clearly change success, and **a pattern always present in the demonstrations won over the `/subtask` instruction** (the arm went to the closest cube).

### Why this rig can test it

This rig already has the failure the method is for: on the 21-task cubes-and-bowls data, SmolVLA picks the *named* cube of two only 2/6 times. So the question is concrete: does System 2 supervision make the instruction win, and is a `/subtask` that contradicts the demonstrated habit obeyed?

### Bars (frozen in the prompt before any number; WS-062)

- A1: ≥ 70 % of grasp/release subtask boundaries from `lerobot-annotate` within ±0.5 s of a gripper event on a 20-episode pilot (< 50 % → stop: the annotator is not usable here). An instrument check on the labeller before anything is trained on its labels — the "audit your labels where they matter" finding, applied before the labels are used.
- A2: G0.5 zero-shot chunk ≤ 667 ms on the Orin at MAXN (> 1,500 ms → arm gates dropped). The tick budget first, as in chapter 11b.
- A3 / B1: the task-swap probe (the TurboVLA instrument: same frame, two instructions, max |Δ| over 10 actions, degrees), zero-shot and then two LoRA runs identical except the recipe — with vs without subtasks — one factor; B1 needs with − without ≥ +2° and with ≥ 5°.
- B2: no regression in held-out chunk error (with ≤ without × 1.05).
- C1: named cube on the arm, 10 sealed layouts committed before any trial, 20 trials per checkpoint; pass ≥ 14/20 against the recorded 2/6.
- C2: the habit test — `/subtask` names cube Y where the policy's own choice was X; obeys ≥ 7/10, and ≤ 3/10 means the blog's finding reproduces on this rig.
- Verdict map, written in advance (chapter 4.5a): reproduces if B1 and C1 pass; fail if either fails with no confound; else not proven. C2 is its own claim row ("habit > instruction").

### Status

Started 2026-10-30, scripts committed before any number; annotation running. Nothing to report yet. When it lands, this section gets the numbers and 11a.9 a row.

## 11a.9 How this family evaluates, side by side

| Paper | Platform | n per cell | Seeds / intervals | Matched control | Noise floor measured |
|---|---|---|---|---|---|
| SmolVLA | LIBERO, Meta-World, SO-100/101 | 10 trials/task sim; real n unstated | none | π0 fine-tuned on same data; ACT | no |
| TurboVLA | LIBERO, RoboTwin, Piper | 50 / 100 rollouts/task; real 40/task | none | π0.5 same data/protocol | no |
| VLA-JEPA | LIBERO(+Plus), SimplerEnv, Franka | 500 eps/suite; real in appendix | none | **w/o human video row under every table** | no |
| REKI | LIBERO-Plus, RoboTwin, RoboCasa | App. Table 3; oscillation 20 matched seeds/task | per-subset ticks | same checkpoint with/without memory at eval | no |
| RTC | Kinetix sim; π0.5 bimanual | **2,048 rollouts/point**; real 10/task/delay | **Wilson 95 % (sim); ±SEM (real)** | hard vs soft mask; BID; TE | no |
| VLAs-as-Tools | LIBERO-Long, RoboTwin, CALVIN | App. D | none | direct-planner row; interface-only row | no |
| Augmentation | LIBERO-Spatial/Plus | 500 IND; 1,662 OOD | 1 seed/config, stated | no-aug PPO | **yes: 0.06 / 1.77** |
| SO-101 bench | SO-101 real | **20 trials/model-task** | none | four policies same data | no |

Read with chapters 3–7:

- Real-robot n in this family: 10 (SmolVLA speed test, RTC), 20 (SO-101 bench), 40 (TurboVLA), unstated (SmolVLA tables, VLA-JEPA). Nobody reports intervals on hardware except RTC's SEM. Your episode bootstrap on 69 rollouts is stricter than every paper that runs your arm.
- The two protocol ideas to steal: the augmentation paper's **noise floor** (same checkpoint, full suite twice) and VLA-JEPA's **"w/o X" row under every table**. Both are one extra run.
- The pattern across TurboVLA Table 3, VLA-JEPA Table 3 and SmolVLA Table 5 is that the biggest numbers in each paper come from a *data* or *pretraining* ablation, not from architecture: language on/off (−27), human video on/off on perturbations (−17), community pretraining on/off (−27). The architecture ablations are 1–6 points at n where 1–6 points is noise. That is the reason chapter 3 says the baseline is the method: on this hardware the data decides, and the first gate on any new policy is the data-side control.
- Only VLAs-as-Tools and REKI name a *mechanism* for their gain (invocation shift; perceptual aliasing) and test it with a designed measurement (Table 1's direct-planner row; the oscillation criterion over matched seeds). Those are the two to imitate when writing the TurboVLA correction and the G14 post.

## 11a.10 Corrections and additions to chapter 11 and the register

- 11.4 (boundaries): the prefix overwrite is RTC's hard mask (Fig. 5, below soft masking at small d); add the PyTorch soft-mask comparison on the 254 boundaries as the next gate, before any ggml backward work; adopt throughput as the result metric and the jump as the instrument check.
- Chapter 8 / TurboVLA correction row: cite Table 3 (LIBERO-Goal 97.4 → 11.6 without language) as the demonstration that the architecture reads language when the data requires it; your fact 7 is a statement about the data, and the port divergence is a separate, now-tested (path A) statement about the model.
- G14 post: adopt REKI's oscillation criterion (reversal at ≥ 3 consecutive chunks) and report the number under it; cite 18.6 → 4.1 % with one past frame as the external mechanism.
- Register per-episode record: add **recovery opportunity / recovery success** (SO-101 bench §2.5.4) alongside the verdict; it costs nothing with the labels you already take.
- Chapter 6 (instrument): add the **noise-floor replay** — same checkpoint, same episode manifest, twice — as a standing check before any gate on a learned policy.
- Harness G24: unit-level fine-tuning (TAPT component i) on the 69 rollouts segmented into the harness's actions is the first experiment; expect a policy evaluated as whole-episode to lose points when called as units (VLAs-as-Tools Table 1), and write the bar on unit success.
- SmolVLA plugin check: state into the VLM prefix (Table 11), re-observe every ≤ 10 steps (Table 13), 10 Euler steps; record g (refill threshold) in every protocol.

## 11a.11 Exercises

1. From SmolVLA Table 13 and your 238 ms chunk, compute the expected success loss from executing a full 50-step chunk versus re-observing every 10, and state what it predicts for G-boundary's two arms.
2. Write TurboVLA's four equations (4–9) from memory, then annotate each with the line of your port that implemented it and whether it matched. That table is the correction row.
3. Implement REKI's oscillation criterion over your 69 rollouts and the G14 episodes; report the rate; say whether 24 % and their 18.6 % measure the same thing.
4. Implement RTC's soft mask (eq. 5) in PyTorch for SmolVLA on the Orin, offline; run the 254 boundaries with shared noise; report the jump distribution paired against the prefix overwrite. Do not touch vla.cpp until this is done.
5. Run the noise-floor replay: one SmolVLA checkpoint, the same 69-episode manifest, twice, fixed seeds; report the absolute difference in success and in G9c-3 verdicts. Put the number in chapter 6.
6. Compute recovery rate (SO-101 bench definition) for the 69 rollouts from the existing labels; decide whether it separates your policies where success does not.
