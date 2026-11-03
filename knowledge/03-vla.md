# 03. Vision-language-action models (VLAs)

Book depth: chapter 11 (BC, chunking, flow heads, boundaries), 11a (SmolVLA, VLA-JEPA,
REKI, RTC, SO-101 benchmark), 11b (chunking and edge cost).

## The landscape in one table

| Model | Backbone | Action output | Key idea |
|---|---|---|---|
| ACT (2023) | ResNet + transformer, no language | chunk of continuous actions, CVAE | chunking + temporal ensembling; strong with ~50 demos |
| Diffusion Policy (2023) | CNN/transformer | chunk, DDPM denoising | multimodal actions |
| RT-2 / OpenVLA | VLM (PaLI / Llama 2 + SigLIP+DINOv2) | discrete tokens, 256 bins per dim, autoregressive | reuse LLM decoding as-is |
| π0 / π0.5 (Physical Intelligence) | PaliGemma VLM | action expert, flow matching, chunk of 50 | continuous high-rate control; π0.5 adds co-training on web/semantic data for open-world generalization |
| FAST tokenizer | any autoregressive VLA | DCT-compressed action tokens | makes autoregressive VLAs competitive at high frequency |
| SmolVLA (HF, 2025) | SmolVLM, upper layers skipped, 64 visual tokens/frame | flow-matching action expert, interleaved cross/self attention | ~0.45B, trains on 1 GPU, community SO-100/101 data, async inference |
| VLA-JEPA (2026) | VLM + JEPA pretraining on video | action head fine-tuned after | latent future prediction without leakage as pretraining |
| GR00T N1 (NVIDIA) | VLM "System 2" + diffusion transformer "System 1" | continuous | dual system: slow reasoning, fast motor |

## Mechanism: anatomy of a modern VLA (SmolVLA / π0 style)

```
cameras -> vision encoder -> visual tokens ┐
instruction -> text tokens ────────────────┼-> VLM layers -> features (prefix, cached)
robot state -> linear -> 1 token ──────────┘                   │ K/V
noisy action chunk A^τ (H×D) + τ -> action expert ──── cross-attn ┘ + causal self-attn
                                      -> velocity v(A^τ, τ | obs)  -> integrate 10 steps -> chunk
```
- **Chunk**: predict H future actions at once (H = 50 at 30 Hz ≈ 1.7 s).
- **Flow matching head**: learns a velocity field from noise to actions (04). Multimodal,
  continuous, ~10 integration steps.
- **Execution**: run part of the chunk (e.g. 25 of 50 steps), then replan.

## The problems that interviewers love

**Compounding error (BC).** Small errors take the robot off the demo distribution, where
errors grow. Mitigations: chunking (fewer decisions), DAgger-style corrections, noise
injection during demos, data at recovery states.

**Multimodality.** MSE averages modes → drives into the obstacle. Fix: generative heads
(diffusion, flow, CVAE) or discrete tokens (a categorical over bins is multimodal).

**Chunk boundaries.** Each new chunk is sampled independently: at the switch the robot can
jump between modes → jerk. Fixes:
- *Temporal ensembling* (ACT): predict a chunk every step, average overlapping predictions
  with exponential weights. Smooth, but averaging modes again, and needs inference every step.
- *Real-Time Chunking* (RTC, Physical Intelligence): generate the next chunk while executing
  the current one; the actions that will execute during inference are "frozen" and the rest
  is inpainted to be consistent with them. No retraining; works for diffusion/flow policies.
  Robust to delays over 300 ms in their tests.
- *Async inference* (SmolVLA): decouple prediction from execution; request a new chunk
  when the queue drops below a threshold.

**Latency budget.** Control at 30 Hz = 33 ms per step. A VLA call of 300 ms means ~10
steps of motion happen during one inference: either the robot pauses (synchronous) or you
need async/RTC. Your edge-inference paper is about this budget on an Orin.

**Action representation.** Absolute joints (LeRobot default) vs deltas vs end-effector
pose; normalization per dim; gripper as continuous vs binary.

## Questions

1. Why do VLAs predict chunks? Give three reasons.
<details><summary>Answer</summary>(1) Fewer decisions per episode → less compounding error. (2) One expensive inference covers many control steps → feasible at high control rates. (3) Committing to one mode for a chunk avoids flickering between modes; and chunks capture non-Markovian behavior (pauses, operator idiosyncrasies) better.</details>

2. Discrete action tokens (RT-2/OpenVLA) vs flow-matching expert (π0/SmolVLA)?
<details><summary>Answer</summary>Tokens: reuse the LLM decoder unchanged, multimodal via a categorical, but autoregressive over H×D tokens is slow and 256 bins limit precision (FAST compresses this). Flow expert: continuous precision, whole chunk in ~10 parallel steps, needs an extra module and a different loss. At high frequency and dexterity, continuous heads have won so far.</details>

3. Explain temporal ensembling vs RTC in one minute each.
<details><summary>Answer</summary>Temporal ensembling: query every step, keep all overlapping chunk predictions for the current time, average with weights exp(−m·i). Smooth but expensive and it blurs modes. RTC: while executing chunk k, generate chunk k+1 conditioned on the actions of chunk k that will still be executed during the inference delay (frozen), inpainting the rest with guidance so it continues smoothly; the boundary is consistent without averaging and without pauses.</details>

4. Your policy runs at 300 ms per inference on the Orin, control at 30 Hz, chunk 50, you execute 25. What happens at each boundary with synchronous execution? Fix?
<details><summary>Answer</summary>After 25 steps (~830 ms) the robot stops ~300 ms waiting for the next chunk (≈9 control periods), then starts a chunk computed from an observation that is already stale. Pauses, jerks, and in dynamic tasks failure. Fix: async inference (start the next prediction when ~10 actions remain), and RTC-style consistency for the boundary; or reduce latency (fewer denoising steps, smaller VLM, quantization, vla.cpp).</details>

5. What does VLA-JEPA change compared to "latent action" pretraining?
<details><summary>Answer</summary>Latent-action methods learn an action variable from pixel differences between frames; they pick up nuisance motion (camera shake, background) and can leak future information into the input. VLA-JEPA predicts the latent of future frames from the current observation only, with a target encoder that sees the future solely as a training target (leakage-free), and predicts in latent space, not pixels. Then a simple action-head fine-tune. Book 11a.3.</details>

6. Why might a 0.45B SmolVLA match a 3B+ VLA on SO-101 tasks?
<details><summary>Answer</summary>The tasks are narrow and in-distribution; most of the capacity of a large VLA buys open-world generalization and language breadth that a single-scene benchmark does not test. Also SmolVLA was trained on community SO-100/101 data (matching embodiment). Your own 9/10 for both VLA-JEPA and SmolVLA on the bowls task is the same story: the benchmark did not separate them.</details>

7. Interviewer: "How would you get a VLA to recover from failures?"
<details><summary>Answer</summary>Data: record recoveries (DAgger, intervention data, failure-then-recover episodes). Detection: a monitor that flags failure early (world-model surprise, progress estimate, action-uncertainty) and triggers a retry or a different sub-skill. Structure: a high-level planner (VLM/LLM) that re-issues sub-tasks. Then RL fine-tuning on the real robot if a reward exists. Cite your harness: verifier slot, failure classes, retry budgets.</details>

8. What breaks when you move a VLA from 1 camera to 3 cameras?
<details><summary>Answer</summary>Token count and latency (~3× tokens, attention cost more); calibration and camera ordering must match training exactly; the model may rely on one camera (check with ablations: blank each camera); timestamps must be synchronized. Your "wrist blank" experiment is the ablation to cite.</details>
