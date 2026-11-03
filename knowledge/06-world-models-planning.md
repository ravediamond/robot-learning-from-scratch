# 06. World models, planning, and the world model as evaluator

Book depth: 12 (three roles), 12b (JEPA planning and grounding), 12d (video world models
vs JEPA), 13 (judges, monitors). Coding list item 23 (latent MPC with CEM).

## What a world model is

A model of `p(next state | state, action)`. "State" can be pixels, tokens, or a learned
latent. Three uses, very different in what they demand:

| Role | Question it answers | Needs |
|---|---|---|
| **Planner** (MPC) | Which action sequence reaches the goal? | Accurate multi-step prediction under ACTION CHANGES (counterfactuals) |
| **Evaluator / monitor** | Is this rollout going to fail? Is this chunk good? | Calibrated surprise or progress signal; must have seen failures |
| **Simulator / data source** | Generate training experience or evaluate policies offline | Visual fidelity + physics + action-following over long horizons |

Your register's lesson in one sentence: a model can be good at one role and useless at
another (predicts well at rest, fails as an evaluator). Never claim "we have a world model"
without saying which role it passed, against which baseline.

## The families

- **Latent dynamics, reconstruction-based**: World Models (Ha & Schmidhuber), **Dreamer**
  (RSSM: deterministic GRU state + stochastic latent; trained with reconstruction + KL; the
  policy is learned by imagining rollouts). Strong in sim RL.
- **Latent, non-reconstructive (JEPA)**: DINO-WM, LeWM, V-JEPA 2-AC. Predict embeddings;
  plan by optimizing actions so the predicted embedding matches the goal embedding.
- **Video / token generative**: Genie-style and large video models (Cosmos, etc.), "world
  action models" that predict video and actions jointly. High fidelity, expensive; can serve
  as policy backbones or simulators. Book 12d compares five ways a video model becomes a policy.

## Planning in latent space (MPC with CEM)

```
z0 = enc(obs), zg = enc(goal_image)
μ, σ = zeros(H, A), ones(H, A)
repeat I iterations:
    sample N action sequences ~ N(μ, σ)              (N, H, A)
    roll out: z_{t+1} = pred(z_t, a_t)  for t < H     autoregressive, batched over N
    cost = ||z_H − zg||²   (or sum over t)
    elites = top-k lowest cost; μ, σ = mean, std of elites
execute the first action(s) of μ, observe, replan       (receding horizon)
```
- **CEM**: robust, gradient-free, parallel. **MPPI**: soft-weighted average of all samples
  by exp(−cost/λ). **Gradient-based**: backprop through the predictor to the actions;
  fast but falls into adversarial/local minima of the model.
- **Failure modes**: compounding prediction error over H; the planner exploits model errors
  (finds actions the model THINKS are great); latent distance ≠ task progress (a goal
  embedding close in L2 may be the wrong object pose). The "planning limits of JEPA" papers
  in your KB study exactly this (book 12b).
- **Cost**: N × H × I predictor calls per step: why LeWM's small latent matters (48× faster
  than DINO-WM).

## World model as evaluator (your H3 / monitor line)

- **Surprise**: error between predicted and observed next latent. Spikes on implausible or
  unexpected events → candidate failure signal.
- **Progress / reward**: distance to goal embedding over time, or a learned progress head
  (Robometer-style reward models trained from trajectory comparisons).
- **Evaluate before executing**: score candidate chunks by imagined outcome. You tested best-of-N
  reranking (H0) and it was a no-go: candidates did not differ enough (oracle/mean error 0.76
  vs bar 0.7), and averaging the 8 chunks already got 45% of the oracle gain with no model.
  Great interview story: the cheapest baseline killed the idea in one day.
- **Calibration**: a monitor is only useful at a chosen false-alarm rate. Report AUROC with
  CI, then TPR at a fixed FPR, and detection time before failure. Compare against the
  cheapest baseline: **episode length / elapsed time** (your failure-monitors paper found
  this baseline is hard to beat).

## Questions

1. Planner vs evaluator: why can a world model pass one and fail the other?
<details><summary>Answer</summary>A planner needs correct counterfactual predictions for actions the data never took; an evaluator needs a calibrated signal that separates success from failure on the policy's own rollouts, which requires having seen failures. Trained on clean demos, a model can predict demo-like futures well (good at rest) yet have no discriminative signal for failures, or vice versa.</details>

2. Write CEM latent MPC from memory (coding list 23). What are N, H, k, I typically?
<details><summary>Answer</summary>See the block above. Typical: N = 300–1000 samples, H = 5–16 steps, k = 10–30 elites, I = 3–10 iterations, execute 1–5 actions then replan. Cost per control step = N·H·I predictor calls (batched).</details>

3. Why does the planner "exploit" the world model, and how do you mitigate it?
<details><summary>Answer</summary>The optimizer searches for actions with the best predicted cost; where the model is wrong, it is often wrong optimistically, and the search finds those spots. Mitigations: short horizons + replanning, ensembles and penalize disagreement (uncertainty), constrain actions near the data distribution (sample around a policy's proposal), regularize the latent.</details>

4. RSSM in one minute.
<details><summary>Answer</summary>State = deterministic h_t (GRU) + stochastic z_t. h_t = GRU(h_{t−1}, z_{t−1}, a_{t−1}); prior p(z_t | h_t); posterior q(z_t | h_t, o_t). Train with reconstruction (and reward) losses + KL(posterior ‖ prior). At imagination time, roll forward with the prior only. Dreamer learns actor and critic on these imagined trajectories.</details>

5. JEPA world model vs video generative model as a robot backbone?
<details><summary>Answer</summary>JEPA: compact latent, cheap rollouts, fits edge hardware, but not human-checkable and may drop task-relevant detail; evaluation by probes and planning success. Video model: rich, checkable, can serve as a simulator and a strong pretraining source from internet video; very expensive to train and run, and visual realism does not guarantee physical or action-faithful predictions. For a SO-101 + Orin: JEPA-scale is the realistic one (book 12d.6).</details>

6. You want a failure monitor. What is your cheapest baseline and your evaluation protocol?
<details><summary>Answer</summary>Baselines: elapsed time / episode length, action-chunk variance across samples, policy's own predictor error. Protocol: label failure onset on N held-out episodes (episode-level split), pre-register the metric (AUROC with episode bootstrap CI, TPR at 5% FPR, lead time), compare paired against the baselines on the same episodes, then a live test where an alert triggers a retry and success rate is compared.</details>

7. Interviewer: "Why not plan directly with a big video model?"
<details><summary>Answer</summary>Latency and cost: sampling N×H video frames per control step is orders of magnitude too slow. Action-faithfulness: video models often ignore small action differences (the counterfactual problem). Current practice uses them to pretrain policies or as offline simulators/evaluators, not as inner-loop planners at control rate.</details>
