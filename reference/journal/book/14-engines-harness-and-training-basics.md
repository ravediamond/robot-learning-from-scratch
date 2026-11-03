# Chapter 14. Engines, the harness, and the training basics you are missing

Three topics an interviewer for a world-models RE role will probe that the gates touch only partly: how inference engines work and why they matter; what the agentic harness is as an experimental apparatus; and the distributed-training vocabulary you have not used yet and must be able to speak.

## 14.1 Inference engines on the edge

### 14.1.1 Why the engine matters more than the model here

SmolVLA: 974 ms in PyTorch on the Orin, 390 ms in vla.cpp, 238 ms with flash attention. Same weights, same task success. The 4× came from: no Python dispatch overhead per op; fused kernels (flash attention computes attention without materialising the score matrix); weights stored in the layout the kernels want (GGUF, bf16); no autograd bookkeeping; a single compiled graph per chunk instead of thousands of small launches. On a GPU with 1 024 CUDA cores and shared memory bandwidth, launch overhead and memory traffic dominate; a compiled engine removes most of both.

The general lesson, said plainly: on edge hardware the first 2–4× is engineering, not research, and it is prerequisite to any research on the control loop (a policy that pauses 974 ms per chunk cannot be evaluated for smoothness).

### 14.1.2 What vla.cpp is

A llama.cpp-style C++ engine over ggml: a tensor library with a static compute graph, backends (CUDA, Metal, CPU SIMD), and a quantised weight format (GGUF). Each policy architecture is a hand-written graph (`src/models/smolvla.cpp` etc.): the VLM's vision tower, the text stack, the action expert's denoising loop unrolled as one forward graph. Adding an architecture means writing its graph against ggml ops (the ACT port, #38: ResNet via `im2col`, frozen batch norm folded into conv weights, the ACT-style decoder reused from TurboVLA's). Things ggml lacks constrain what is possible: no backward rule for flash attention or `concat`, hence no exact RTC guidance in vla.cpp without new kernels (`projects/vlacpp/README.md` §7 lists exactly which rules are missing and what each would need).

Parity (chapter 6) is how a port is trusted: `vla_predict_check` against PyTorch on fixed inputs with shared noise; the SmolVLA GGUF at 0.80° median max |Δ| against a 1° line.

### 14.1.3 The numbers an interviewer may ask for

Twelve VLAs on the Orin NX (your benchmark, merged as vla.cpp #36), minimum ms at defaults: Octo 27, TurboVLA 80, VLA-JEPA 177, GR00T N1.7 251, BitVLA 269, SmolVLA 411, π0 740, Evo-1 893. AGX Orin roughly halves these; Orin Nano Super is ~15 % slower. The template protocol: 3 warm-ups, 20 reps, best of 3 processes, flag screen, re-run of the fastest set, sha256 of every GGUF checked. Thermal: 1 162–1 172 MHz, Tj ≤ 83 °C, no throttling.

### 14.1.4 Precision and memory

bf16 weights halve memory and are what the tensor cores want; f32 matmul (`--mm-prec f32`) was used for the parity check because the GGUF-vs-PyTorch comparison needs to separate implementation differences from precision. On the Orin, peak RSS from `getrusage` excludes CUDA buffers (unified memory), so the vla.cpp memory column was reported with a whole-board `tegrastats` column alongside and the maintainer dropped memory from the table. Know why.

## 14.2 The harness as an experimental apparatus

### 14.2.1 What it is

v1.4–v1.5: the station is a harness; Claude Code is its operator over MCP tools; scripted picks and SmolVLA are interchangeable executors behind `do_action` and `run_policy`; the verifier is a per-task plugin with a credibility tier; every trial is a LeRobot-v3 record with task string, layout id, executor version, verdict and evidence.

Read it not as a product but as a lab instrument. What it gives experiments that a bare `lerobot-rollout` does not:

- **Idempotent dispatch** (`key = (run, action_id, attempt)`): a retried call returns the existing trial; the arm never moves twice for one logical action. Without it, every lost tool response is a potential double pick and an uncounted trial.
- **Two-field results** (`executor_status` × `verdict`): the executor cannot grade itself; the verifier never reads the agent's text. Without it, the agent's plan leaks into the verdict.
- **Budgets per failure class** that the agent cannot raise: a bounded number of re-observes, undos, executor switches. Without it, an autonomous loop escalates its own retries.
- **The verifier card**: per gate, FP / FN against human labels, ambiguous rate, calibration; tier 3 never decides alone.
- **The trace**: agent trajectory + nested spans (skills, tools, perception, executors, VLA, verifier) linked to the robot trajectory; failure-review reads keyframes at action boundaries, not dense video.
- **The flywheel**: `demos/` as a view over tier-1/2 satisfied trials; session planning from failed layouts; G24 as an adoption protocol (a policy replaces a scripted executor only with enough verified demos, a verification window with fallback, and non-inferiority).

The reason it is designed this way is one number: HexaShell's 739 s of agent time for 26 s of robot time. The agent stays out of the motion loop; it dispatches unit actions and reads results at rest.

### 14.2.2 What is and is not run

Specified in v1.4/v1.5; 284 tests; first LEGO teleop session (60 episodes) and snapshot sessions; nothing end to end on the arm. The honest interview line: "designed, unit-tested, not yet validated on hardware; the gates for it (H1, G22, K1, G24) are frozen and waiting for the robot day."

## 14.3 Training basics you have not done, and must be able to discuss

The World Models posting says "profile, debug and scale distributed training runs". You have trained on one GPU at a time. These are the concepts, at the level of being able to reason about them and to say what you would do first.

### 14.3.1 Effective batch and gradient accumulation

The gradient step sees a batch; if the batch does not fit in memory, run k micro-batches, accumulate gradients, step once: effective batch = k × micro-batch. Learning-rate scaling: linear with batch is the usual starting point for SGD-like regimes; for Adam-family optimisers the dependence is weaker and is tuned. TurboVLA path A trains at batch 64 where the paper used 128 effective; the honest note is "half the effective batch, same steps, so fewer samples seen; a deviation named in the protocol".

### 14.3.2 Mixed precision

bf16 activations and weights with f32 master weights and f32 accumulation in the optimiser. bf16 has the exponent range of f32 and 8 bits of mantissa, so no loss scaling is needed (unlike fp16); the price is precision in small gradient updates, which the f32 master copy absorbs. Your G0 ran bf16-mixed, which is why V100s (no native bf16) were excluded from the vast offers.

### 14.3.3 Data parallelism: DDP and FSDP

**DDP**: every GPU holds a full model copy; each processes its own micro-batch; gradients are all-reduced (summed across GPUs) before the step. Scales until the model no longer fits on one GPU. **FSDP** (fully sharded): parameters, gradients and optimiser state are sharded across GPUs; each layer's full parameters are gathered just before its forward/backward and released after. Fits models larger than one GPU at the cost of more communication. The TurboVLA plugin declares `_fsdp_wrap_modules` per transformer layer, which is the unit FSDP shards and gathers.

What breaks at 8 GPUs that worked at 1: the data loader becomes the bottleneck (each GPU needs its own decoded frames; video decoding on CPU saturates; the path A launcher capped workers on a 128-CPU host because too many workers thrashed); random seeds and data order must be per-rank; batch-norm statistics (ACT's ResNet) need syncing or freezing; checkpoint saving must be from one rank or sharded; and any `print` or file write that assumes one process writes k copies.

### 14.3.4 Gradient checkpointing

Trade compute for memory: do not store activations for backward; recompute them from checkpoints during backward. ~30 % more compute, large memory saving. It is how a 24 GB GPU trains a model that "needs" 40 GB; batch-128 OOM on the RTX PRO 4000 in path A's pilot is the kind of failure it addresses.

### 14.3.5 Profiling

`torch.profiler` for op-level time on GPU and CPU, with the trace viewer; `nsys` for the whole process including kernel launches and memory copies; `tegrastats` on the Jetson for GPU/EMC load and power. The first question in any slow run: is the GPU waiting (data loader, CPU preprocessing, synchronisation) or working? Your REAP latency work measured exactly this split (data wait vs GPU step) on the Orin. On a rented box, GPU utilisation under 80 % during training means the loader is the problem.

### 14.3.6 Cost estimation

Steps × seconds-per-step × hourly rate, with seconds-per-step measured in a pilot at the real batch size on the real GPU, plus setup, plus a safety deadline. The path A launcher does this: pilot at batch 64 → 0.78 s/step → 35k steps ≈ 7.6 h → $2.15 at $0.256/h, with a hard deadline so a hung job cannot exceed the cap. Say it this way: "I pilot, measure, then commit a budget with a kill switch."

### 14.3.7 Reproducibility in training

Pin: seeds (and per-rank seeds), data order, library versions, the checkpoint-saving format, EMA if used (the paper's recipe has EMA 0.999; your v2 run matched it), precision. Record the launch command in the protocol. Resume from checkpoint must reproduce the loss curve; if it does not, the pipeline is not deterministic and every comparison across runs is suspect.

### 14.3.8 What you would say

"I have not run multi-node. I have run single-GPU jobs with pilots, budgets and deadlines, and I know what FSDP shards, what breaks at scale (loaders, seeds, BN, checkpoint writes), and where to look first (GPU utilisation, then the profiler). On day one I would ask to shadow one of your distributed runs and reproduce its loss curve on a smaller slice."

## 14.4 Exercises

1. Reproduce the SmolVLA PyTorch vs vla.cpp latency on the Orin and use `nsys` or `torch.profiler` on the PyTorch side to say where the 974 ms goes (launch overhead vs kernels vs memory). One page.
2. Read vla.cpp's `src/models/smolvla.cpp` denoising loop and write, in pseudocode, the graph it builds for one Euler step. Say which op would need a backward rule for exact RTC.
3. Write a 30-line DDP training loop for a toy model with gradient accumulation and per-rank seeds; run it with 2 processes on CPU; verify the accumulated gradient equals the single-process gradient at the same effective batch.
4. From the path A pilot numbers, recompute the cost estimate and the deadline; then do it for a hypothetical 4-GPU DDP run at the same per-GPU throughput.
