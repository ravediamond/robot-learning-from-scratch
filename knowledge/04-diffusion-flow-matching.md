# 04. Diffusion and flow matching for actions

Book depth: 11.3 (flow matching for action heads), P2.15. Coding list items 11, 12.

## Why generative heads

Demonstrations are multimodal (left or right around the cup; grasp the rim here or
there). An MSE head predicts the conditional MEAN, which can be an action no operator ever
took. A generative head learns the conditional DISTRIBUTION p(action chunk | observation)
and samples one mode.

## DDPM in one page

**Forward (fixed):** add Gaussian noise over K steps.
`x_k = √ᾱ_k · x_0 + √(1 − ᾱ_k) · ε`, ε ~ N(0, I), ᾱ_k decreasing from ~1 to ~0.

**Training:** pick k at random, noise the clean action chunk x_0, train a network to
predict the noise:
`L = || ε − ε_θ(x_k, k, obs) ||²`

**Sampling:** start from x_K ~ N(0, I), repeatedly remove predicted noise (plus a bit of
fresh noise) for K steps (Diffusion Policy: 100 train steps, ~10–16 with DDIM at inference).

Intuition: predicting ε is equivalent to learning the **score** ∇ log p(x_k), the direction
toward higher-probability actions at that noise level.

## Flow matching in one page (π0, SmolVLA)

Define a straight path from noise x₀ ~ N(0, I) to data x₁ (the true chunk):
`x_t = (1 − t) · x₀ + t · x₁`,  t ∈ [0, 1]
Its velocity is constant: `dx_t/dt = x₁ − x₀`.

**Training:** sample t, build x_t, regress the velocity:
`L = || v_θ(x_t, t, obs) − (x₁ − x₀) ||²`

**Sampling:** start at noise, integrate with Euler: `x ← x + Δt · v_θ(x, t, obs)` for ~10 steps.

Why people prefer it for actions: simpler (no noise schedule maths), straight paths need
few integration steps, same multimodality. (Convention warning: papers and codebases disagree on
whether t = 0 or t = 1 is noise, sometimes a paper and its own code differ. Read the code.)

```python
# training step, 6 lines (coding list item 11)
x1 = actions                                  # (B, H, D) normalized
x0 = torch.randn_like(x1)
t = torch.rand(B, 1, 1)
xt = (1 - t) * x0 + t * x1
loss = ((model(xt, t.view(B), obs) - (x1 - x0)) ** 2).mean()
```

**Number of steps vs quality.** 1 Euler step = predicting the mean velocity → back to
MSE-like averaging. More steps → curved trajectories followed better. ~10 is the standard
VLA trade-off; it is also the latency knob you can turn on edge hardware.

**Classifier-free guidance (CFG).** Train with the condition dropped some of the time;
at sampling, `v = v_uncond + w · (v_cond − v_uncond)`. Sharpens adherence to the condition
(e.g. the instruction). Costs 2 forward passes per step.

## CVAE (ACT)
Encoder sees (obs, true chunk) → latent z; decoder (obs, z) → chunk. KL keeps z ~ N(0, I).
At test time z = 0 (the mean): a "style" variable, deterministic output. Simpler than
diffusion, works well at ~50 demos, less expressive for strongly multimodal data.

## Questions

1. Draw why MSE fails on a 2-mode action distribution, with numbers.
<details><summary>Answer</summary>Half the demos go to x = −1, half to x = +1. The MSE minimizer is the mean, x = 0: exactly between the modes, which no demo did (the obstacle). A generative head samples ≈ −1 or ≈ +1. Week 4's Friday lab measures this.</details>

2. Write the flow-matching training step and Euler sampler from memory.
<details><summary>Answer</summary>See the block above for training. Sampler: `x = randn(B, H, D); for i in range(N): t = i / N; x = x + (1/N) * model(x, t, obs)`. Return x, then un-normalize. Fix the noise seed when comparing runs (parity).</details>

3. ε-prediction (DDPM) vs velocity (flow): what does each network learn, and how are they related?
<details><summary>Answer</summary>DDPM's network predicts the noise added (equivalently the score at that noise level). Flow matching predicts the velocity of a deterministic path from noise to data. For Gaussian paths they are linear reparameterizations of each other (v-prediction is a mix of ε and x₀). Practical difference: flow with straight paths and simple Euler; diffusion with a variance schedule and DDPM/DDIM samplers.</details>

4. Why does 1-step sampling of a flow policy behave like an MSE policy?
<details><summary>Answer</summary>At t = 0 the network sees pure noise, essentially uninformative about which mode, so the best velocity in L2 points toward the mean of the data given obs. One Euler step lands on that average. More steps let the trajectory commit to a mode as x_t gains information.</details>

5. How do you reduce inference latency of a flow-matching VLA on an Orin without retraining?
<details><summary>Answer</summary>Fewer integration steps (10 → 5, check success), cache the VLM prefix K/V across steps, run the VLM less often than the expert, lower precision (fp16/int8), smaller visual token count, a C++ runtime (vla.cpp). Measure success vs latency as a curve, pre-register the acceptable drop. With retraining: distillation to 1–2 steps (consistency / shortcut models).</details>

6. What is classifier-free guidance and would you use it for a VLA?
<details><summary>Answer</summary>Train conditional and unconditional (condition dropped ~10%), extrapolate between them at sampling to strengthen the condition. For a VLA: can increase instruction following, but doubles compute per step and can push actions to extremes (out of distribution). Worth testing only if language following is the bottleneck.</details>

7. Interviewer: "Diffusion policy is slow. Why not just use discrete tokens?"
<details><summary>Answer</summary>Tokens are multimodal and reuse the LLM decoder, but autoregressive decoding of H×D tokens is slow too, and bins cap precision. FAST (DCT compression) narrows the gap. Flow heads generate the whole chunk in parallel in ~10 steps with continuous precision. It is an empirical trade-off by task: high-frequency dexterous control has favored flow; language-heavy generalist training benefits from tokens (π0.5 / FAST co-training).</details>
