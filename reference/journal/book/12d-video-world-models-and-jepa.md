# Chapter 12d. Video world models as policies, and how they differ from JEPA

Chapters 12–12c are about one family of world models: JEPA-style models that predict the future as an encoder's latent vector and never draw a pixel. In 2025–2026 a second family became the most visible route to robot policies: **video world models**, large video-generation networks adapted so that their prediction of the future drives the robot's actions. Many papers now call them **world action models (WAMs)**. A research engineer working on world models will be asked to compare the two, usually in one question: *"Why JEPA and not a video model?"* (or the reverse). This chapter gives the concepts, the main designs with their numbers, what the evidence actually shows, and what any of it means for a one-person lab with a Jetson.

Sources and their limits: Cosmos Policy (2601.16163), Video Prediction Policy 2 (2610.10270), VERA (2605.27817) and Fast-WAM (2603.16666) were read from their abstracts and main text on 2026-10-24, not at the depth of the "a" chapters; LingBot-VA, Fast-WAM and LaWAM from the LeRobot documentation of their integrations (main branch, 2026-10-23); GreenWAM (2610.18774) from the project's notes; LaLa-World and Planning Limits from chapter 12b. Numbers below are the authors' and are mostly on simulation benchmarks that are close to saturation (see 12d.5).

## 12d.1 Two ways to predict the future

Every world model answers the same question — *given what I see now and what I do, what will I see next?* — but it can represent "what I will see" in two different ways.

**Generative (video) world models predict the observation itself.** The output is future frames, usually not raw pixels but the latents of a **variational autoencoder (VAE)** that compresses each frame and can decode it back to pixels. The training objective is to reproduce the future frames (in practice, a diffusion or flow-matching objective over the VAE latents, P1.14). Because the target is the frame, the model has to model *everything* in it: the robot, the objects, but also textures, shadows, reflections, the background. In return you can **look at** its prediction: it renders a video of the imagined future.

**Joint-embedding predictive (JEPA) world models predict an abstract representation.** An encoder maps each frame to a vector; a predictor maps (current vector, action) to the next vector; the loss compares predicted and actual vectors in that space, and nothing is ever decoded. The encoder is free to discard what is unpredictable or irrelevant (the exact texture of the table), which is the whole argument for JEPA (chapter 12.1). The price is the collapse problem (P1.12, 12.2), and that the prediction cannot be looked at without training a decoder or a probe.

A frequent confusion: a video model's VAE latent and a JEPA latent are both "latents", but they are trained for opposite purposes. The VAE latent is trained to **reconstruct** the frame, so it keeps whatever the pixels need. The JEPA latent is trained to be **predictable and informative**, so it keeps whatever helps predict the next representation, and nothing forces it to keep a small detail like the exact gripper opening. That is the root of most of the differences below.

## 12d.2 What a world action model is

A list maintained by the community (awesome-world-action-models) uses one rule: a model is a WAM **when its predicted future is used by the action path**. A world model that only produces video is not a policy; a policy that never predicts anything is not a WAM. It sorts WAMs by what the future looks like:

- **Render-and-decode**: the model produces a visible future (RGB video, depth, multi-view frames) and an action module reads or inverts it.
- **Latent-only**: the future is predicted in hidden states or compact signals, never decoded to video at deployment.
- **Video-generation-free**: the future signal comes from something other than a video generator (an LLM, a JEPA or DINO predictor, a geometric model). JEPA world models used for control (DINO-WM, V-JEPA 2-AC, LeWM, VLA-JEPA) sit here.

## 12d.3 Five ways a video model becomes a policy

**1. Plan in video, then recover the actions (inverse dynamics).** Generate a video of the task being done, then train a separate, small **inverse dynamics model (IDM)** that, given two consecutive frames, outputs the action that moves the robot from one to the other. UniPi (2023) founded the approach. VERA (2605.27817) keeps a 14-billion-parameter video planner frozen and trains a small embodiment-specific IDM that predicts an image-space Jacobian and inverts it from optical flow; it runs closed loop (plan, execute a short chunk, observe, replan). Reported numbers: in simulation, 94 % on a Panda task where a UniPi-style IDM reached 0 %; on real basic tasks with DROID-trained checkpoints, DreamZero 90 %, VERA 60 %, π0.5 30 %. Strength: the video planner is embodiment-agnostic and can be trained on human or internet video; only the IDM is robot-specific. Weakness, by the authors' own account: real-world failures came mostly from the video-to-action step, recovery degrades when the predicted motion is small in pixels, and nothing in RGB tells you force.

**2. Generate video and actions together.** One network predicts future frames *and* the action chunk, so the actions are trained alongside a full model of the scene. GR-1 and GR-2 did this with GPT-style transformers; **Cosmos Policy** (2601.16163) takes NVIDIA's 2-billion-parameter Cosmos-Predict2 video diffusion model and, without changing its architecture, inserts actions, joint states and a value estimate as extra "latent frames" between the image frames, then post-trains it on demonstrations (half of each batch) and rollouts (the other half, for the world-model and value parts). Reported: LIBERO 98.5 % average; RoboCasa 67.1 % over 24 tasks with 50 demonstrations per task; 93.6 % average on four real bimanual ALOHA tasks; and a **planning mode** (propose several candidate chunks, predict each one's future and value, execute the best) that added about 12.5 points of task completion on two hard real tasks, using a planning model fine-tuned on 648 rollouts. **LingBot-VA** (in LeRobot as `lingbot_va`) is an autoregressive version: a ~5-billion-parameter dual-stream transformer on the Wan2.2 video stack predicts video latents (about 20 denoising steps) and actions (about 50) chunk by chunk, feeding real observed frames back into its cache. Its frozen VAE and text encoder add about 20 GB.

**3. Use the video model as a representation, not a renderer.** Keep the video model's internal features and train an action head on them, without generating frames at test time. Video Prediction Policy (VPP) conditions an implicit inverse-dynamics head on a video-diffusion model's predicted representations; VPP2 (2610.10270) adds manipulation-focused video pretraining and a single-step distilled planner, and reports +18.5 points over the strongest baseline on zero-shot real ALOHA tasks. **Fast-WAM** (2603.16666, in LeRobot as `fastwam`, built on Wan2.2-TI2V-5B) asked the sharpest question in this family — *do WAMs need test-time future imagination?* — and answered no: keeping video prediction as a **training-time co-training objective** but skipping it at inference stayed competitive with imagine-then-execute variants, while removing the video co-training cost much more; latency 190 ms, more than 4× faster. (The LeRobot port reports 96.4 % average on LIBERO over 2 000 episodes.) **GreenWAM** (2610.18774) found the same direction from another angle: keeping world-model objectives in the fine-tuning mix (0.05 forward dynamics, 0.05 image-to-video) added 8 points in randomised scenes and nothing in clean ones.

**4. Use the video model to make training data.** Generate many plausible robot videos for new tasks or scenes, label them with an IDM or pseudo-actions, and train an ordinary policy on them (DreamGen). The policy never runs a video model; the video model is a data engine.

**5. Use the video model as an evaluator or simulator.** Run candidate policies inside the learned video simulator instead of on the robot, or score imagined outcomes. This is the cheapest use for evaluation-heavy teams and the hardest to trust, because the simulator's errors are correlated with exactly the situations a policy finds hard. GreenWAM's analysis is the useful version: comparing the policy's imagined frames with the real ones, out of distribution, success fell from 85 % to 27 % as the prediction error rose, and accurate imagination over the first 100 steps doubled the success rate; in distribution, 19 of 20 failures had imagined the outcome correctly (the actions were wrong, not the prediction).

Between the families sits **LaWAM** (in LeRobot as `lawam`): a Qwen3-VL backbone, a *latent* action and world model, and a flow-matching action head; 98.5 % on LIBERO in its LeRobot port, and one of the policies validated on real hardware through LeRobot's remote inference (chapter 14b.10). It is a reminder that "latent" world modelling is not the property of JEPA alone.

## 12d.4 JEPA and video world models side by side

| | Video world model / WAM | JEPA world model |
|---|---|---|
| What is predicted | future frames (VAE latents that decode to video), often with actions | the next encoder vector, given an action |
| Training target | reconstruct the future observation (diffusion / flow matching) | match the target encoder's vector (MSE in latent space) |
| Decoder | yes; you can watch the imagined future | no; needs a probe or a separately trained decoder to inspect |
| Collapse risk | none from the objective (the pixels are a fixed target) | yes; needs stop-gradient + EMA, VICReg, SIGReg or a frozen encoder (P1.12) |
| Where capacity goes | everything visible, including irrelevant detail | whatever helps prediction; may drop small but task-relevant detail |
| Typical size | 2–14 B parameters (Cosmos Policy 2 B; LingBot-VA ~5 B + ~20 GB frozen parts; Fast-WAM on a 5 B backbone; VERA's planner 14 B) | from a few million parameters (the LeWM-style W1 here, 192-d latent) up to about 1 B (LaLa-World's 0.8 B language-model predictor), usually on a frozen or small encoder |
| Training compute | tens of GPUs for days (GreenWAM: 64 H100s) | one GPU, hours (W1: a few dollars) |
| Inference | slow if it imagines (several denoising steps per frame); 190 ms for Fast-WAM without imagination; 35 ms for GreenWAM only after distillation and quantisation on an RTX 4090 | one forward pass per step; cheap enough to run hundreds of rollouts for planning |
| Can use action-free video | yes (captions or language as condition) | yes (LaLa-World: actions as language; video-only pretraining of the encoder) |
| Planning | possible but expensive per candidate (Cosmos Policy scores candidates in parallel across GPUs) | the natural use: CEM over many candidates in latent space (12b) |
| Main evidence for | strong benchmark results; scales with internet video; interpretable imagination | efficiency; matched comparison where it beats a generative model for planning (below) |
| Main open problem | cost and latency; whether test-time imagination is needed at all | grounding: whether the latent keeps what the task needs (12b, 12c) |

**The one matched comparison.** Most of these numbers come from different papers, data, and protocols, and cannot be subtracted. LaLa-World (chapter 12b.8a) is the exception: a JEPA-style latent predictor and a generative video model (RAEv2) with the **same encoder, same language conditions, same training data and the same parameter count**. The latent model planned better (59.4 vs 48.0) after six times less training and planned 13.9 times faster. In simulation only, at horizons of six steps or fewer, one matched study: strong evidence for the efficiency argument, not a verdict on the families.

## 12d.5 What the evidence supports, and what it does not

1. **The value of video is mostly in training, not in imagining at test time.** Fast-WAM's ablation, VPP's design, GreenWAM's co-training result and Cosmos Policy's auxiliary-loss ablation (−1.5 points without them, −3.9 from random weights) all point the same way: predicting the future is a strong *training signal and representation*, while rendering it at inference mostly costs time. This is also the JEPA argument (predict, but in a space that is cheap), arrived at from the other side.
2. **Planning with any world model is limited by horizon.** Planning Limits (12b) showed that even a perfect simulator used as the world model falls from 92 % to 41 % when the goal is beyond the planning horizon. Generative or latent, a world model plans only as far as it rolls out reliably. Cosmos Policy's planning gain came from scoring a few policy-proposed chunks with a value estimate, not from long-horizon search.
3. **Prediction quality is not decision quality.** For both families: one-step fidelity barely correlates with planning success (2610.19375, r ≈ −0.01), decision-relevant error matters more than average error (DRPE), and a model with worse one-step error can plan better (SALT) (all chapter 12c). A beautiful imagined video is a demo, not evidence.
4. **Imagination error can be a failure signal, at scale.** GreenWAM's out-of-distribution result is the strongest positive case for a world model as a monitor; this repository's W1 monitor failed the fresh-rollout transfer (C1: imagined latent 0.70 vs current latent 0.82). The differences are size, data (GreenWAM trained on ~1 675 hours including the policy's own setting) and representation; the data-engine use case is the cheap way to test whether data is the missing piece here.
5. **The benchmarks are saturated.** LIBERO averages of 96–98.5 % (Cosmos Policy 98.5, LaWAM 98.5, Fast-WAM 96.4 in its LeRobot port, LaLa-Policy 97.7) cannot rank these methods; the LeRobot LaWAM page even notes that LIBERO success depends on the MuJoCo version. The informative numbers are the harder ones: RoboCasa with 50 demonstrations, real-robot tasks, out-of-distribution scenes, and anything with a matched baseline (chapter 2a.3).

## 12d.6 What this means on a SO-101 with a Jetson

- **A video WAM does not run on the Orin.** Models of 2–14 billion parameters with tens of denoising steps per prediction are far beyond 16 GB of shared memory and the latency budget of 10a.5; even GreenWAM's 35 ms needed a desktop GPU, distillation and quantisation, and was not validated on onboard computers. The realistic deployment is remote inference from a rented GPU (chapter 14b.10), with all the latency and fault-handling questions that brings. The SmolVLA experience (1.83 s per prediction from a Mac, longer than its 1.67 s horizon) is the warning.
- **Training one is out of reach; fine-tuning one is borderline.** Released checkpoints (Cosmos Policy, LingBot-VA, Fast-WAM, LaWAM in LeRobot) can be fine-tuned on a rented large-memory GPU for tens of dollars per run at small scale; whether 60–600 SO-101 episodes are enough is exactly the kind of claim to test, not assume.
- **The cheap experiments are offline.**
  - *Video features as a representation:* encode the 69 rollouts with a frozen pretrained video model and repeat the C1 probe (outcome from current vs imagined features), against W1 and the same baselines. This tests point 3 of 12d.3 at the cost of one GPU-hour.
  - *Imagination error as a monitor:* run a released WAM's video prediction on recorded rollouts, compute prediction error against the real frames, and score it as a failure detector with the REAP v2 controls (shuffled actions, common horizon, length). This is GreenWAM's analysis on your data, and a direct comparison with the failed W1 monitor.
- **The interview answer** is a comparison, not a camp: video WAMs win benchmarks and use internet video, at a cost in compute and latency that the strongest of them now try to avoid at test time; JEPA models are cheap enough to plan with and to train on one GPU, but have to prove that their latent keeps what the task needs. On a $300 arm the JEPA route was the only one affordable to test, and what was learned (it loses to commanded joints as a planner, it failed as a monitor on teleop-only training) applies to the video route too: the questions are data and evaluation, not the decoder.

## 12d.7 Questions to answer in one minute each

1. What is the difference between a VAE latent in a video model and a JEPA latent?
2. Name the five ways a video model can become a policy, with one example each.
3. What did Fast-WAM test, and what did it find?
4. Why does a video world model not suffer from representation collapse, and why does a JEPA need a defence against it?
5. Give the one matched comparison between a latent and a generative world model, with its numbers and its limits.
6. Cosmos Policy's planning mode: what does it score, and where does the gain come from?
7. Why are LIBERO results not enough to rank these models?
8. Your manager wants a video world model on the robot by next month. What would you measure first, and what would you run instead if the latency does not fit?

## 12d.8 Exercises

1. Draw, for each of the five routes in 12d.3, where the video model sits (training only, inference, data generation, evaluation) and what is learned per robot. Mark which ones could use your SO-101 data alone.
2. From Fast-WAM's paper, find the table that compares imagine-then-execute, no-imagination, and no-video-co-training variants; write the three numbers and the matched control, and say whether the conclusion would survive the 2a.3 tells.
3. Write the protocol for the offline imagination-error monitor of 12d.6 on the 69 rollouts: model, frames, error metric, controls, bar (blank), cost.
4. Prepare the two-minute answer to "JEPA or video world model?" with three numbers, one from each side and one matched comparison, and say what result on your rig would change it.

Sources: [Cosmos Policy, arXiv 2601.16163](https://arxiv.org/html/2601.16163v1); [Video Prediction Policy 2, arXiv 2610.10270](https://arxiv.org/abs/2610.10270v1); [Turning Video Models into Generalist Robot Policies (VERA), arXiv 2605.27817](https://arxiv.org/html/2605.27817v1); [Fast-WAM, arXiv 2603.16666](https://arxiv.org/abs/2603.16666); [awesome-world-action-models](https://github.com/world-action-models/awesome-world-action-models); LeRobot documentation for `fastwam`, `lingbot_va` and `lawam` (main branch, 2026-10-23); GreenWAM notes (project document); LaLa-World and Planning Limits (chapter 12b).
