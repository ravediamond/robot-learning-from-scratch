# 05. JEPA and self-supervised representations

The most likely deep-dive at a world-models startup. Book depth: chapter 12 (collapse,
your two models, three roles), 12a (DINO-WM, LeWM, V-JEPA 2.1, RoboJEPA, H-JEPA…), 12c.

## The idea

**Predict in representation space, not pixel space.** Pixels contain huge amounts of
unpredictable detail (texture, lighting flicker, background motion). A generative model
must spend capacity on all of it. A JEPA (Joint-Embedding Predictive Architecture) encodes
both the context and the target, and predicts the target's EMBEDDING from the context's.
The encoder is free to drop what is unpredictable.

```
context x  -> encoder f  -> s_x  ─┐
                                   predictor g (+ action, + mask position) -> ŝ_y
target  y  -> encoder f' -> s_y  ─┘ loss = d(ŝ_y, s_y)
```
- **I-JEPA**: x = visible image patches, y = masked blocks of the same image.
- **V-JEPA / V-JEPA 2**: same over video (spatio-temporal masks). V-JEPA 2-AC adds an
  action-conditioned predictor for planning. **V-JEPA 2.1** adds a dense loss on all tokens
  (visible and masked) and deep supervision at intermediate layers to get better DENSE
  (spatial) features, which matter for manipulation.
- **World-model JEPA** (DINO-WM, LeWM): x = frame t + action a_t, y = frame t+1.

## Collapse: the central problem

The loss `||g(f(x)) − f'(y)||²` is minimized by f = constant (every input → same vector).
Zero loss, zero information. Three families of fixes:

1. **Asymmetry + stop-gradient + EMA** (BYOL, I-JEPA, V-JEPA): target encoder f' is an
   exponential moving average of f and receives no gradient. Works, but "why" is subtle, and
   it is fragile (EMA rate, predictor capacity).
2. **Variance/covariance regularization** (VICReg): penalize embedding dims whose std over
   the batch falls below 1 (variance term), and decorrelate dims (covariance term). Plus the
   invariance/prediction term. Several weights to tune.
3. **Distribution matching** (LeJEPA's **SIGReg**, used by **LeWM**): push the embedding
   distribution toward an isotropic Gaussian. Project embeddings onto many random directions;
   each 1-D projection should look like N(0, 1) (a normality test statistic such as
   Epps–Pulley as the loss). A collapsed embedding fails the test badly.
   LeWM: end-to-end from pixels, **two loss terms** (next-embedding MSE + SIGReg), no EMA, no
   stop-gradient, no pretrained encoder, ~15M parameters, trainable on one GPU in hours;
   plans up to 48× faster than DINO-WM (which uses frozen DINOv2 patch features).
4. (Alternative) **Frozen pretrained encoder** (DINO-WM): no collapse possible since the
   encoder is not trained; only the predictor learns. Cost: big encoder, features not tuned
   for your dynamics.

## How you know your latent is any good

- **Linear probes**: freeze the encoder, fit a linear map to physical quantities (joint
  angles, object position). Good probe R² = information is there and accessible.
- **Prediction error vs baselines**: the predictor must beat "copy last latent" (the latent
  equivalent of hold-position) by a clear margin at horizon k.
- **Rank / spectrum** of the embedding covariance: effective rank collapsing = partial collapse.
- **Surprise**: prediction error on physically implausible events should spike (LeWM probes
  this). This is the hook for failure monitoring.
- **Planning success** (06): the real test.

## Your evidence (be ready to say these numbers; re-derive them in week 8)

Your register's W1/W2/G-series gates on LeWM-style models on SO-101: what predicted well at
rest, what failed as an evaluator, the receptive-field and surprise findings, the checkpoint
loading audit. Book 12.3–12.4, `reference/journal/papers/jepa-world-model/`.

## Questions

1. Why predict embeddings rather than pixels? Give the strongest counter-argument too.
<details><summary>Answer</summary>Pro: pixel prediction wastes capacity on unpredictable detail and gives blurry averages; latent prediction lets the encoder keep what is predictable and useful, and is much cheaper to roll out for planning. Counter: you lose a grounded, human-checkable output and an objective that cannot collapse; it is hard to know what the latent dropped (maybe exactly the contact detail you need). Video world models (pixel/token generative) are the other camp: book 12d.</details>

2. Explain collapse and three ways to prevent it in 90 seconds.
<details><summary>Answer</summary>Constant embeddings give zero prediction loss. (1) EMA target + stop-gradient + predictor asymmetry (BYOL, I-JEPA). (2) Regularize the batch statistics: keep per-dim variance up, decorrelate dims (VICReg). (3) Match a target distribution, e.g. isotropic Gaussian via random 1-D projections and a normality test (SIGReg, LeWM). Plus: freeze a pretrained encoder (DINO-WM).</details>

3. How does SIGReg detect collapse with random projections?
<details><summary>Answer</summary>If embeddings are isotropic Gaussian, every 1-D projection onto a unit direction is N(0, 1). Sample M random directions, project the batch, compare each empirical 1-D distribution to N(0, 1) with a differentiable test statistic, average. Collapsed (constant) embeddings project to a spike (variance 0), maximally non-Gaussian → large loss. 1-D tests are cheap and avoid estimating a D×D covariance. (Coding list item 15.)</details>

4. DINO-WM vs LeWM: what does each choose, and what does it cost?
<details><summary>Answer</summary>DINO-WM: frozen DINOv2 patch features (strong, general, no collapse risk), trains only a predictor on offline trajectories; heavy encoder, many patch tokens → slow planning. LeWM: small encoder trained end-to-end with SIGReg, compact latent; much faster planning (up to 48× reported), features specialized to the environment; risk: weaker generality, and its latent must be validated (probes, baselines).</details>

5. What does EMA actually do for the target encoder?
<details><summary>Answer</summary>θ' ← m θ' + (1 − m) θ, m ≈ 0.99–0.999. The target changes slowly, giving the online network a stable regression target and preventing the two from co-adapting toward the trivial solution quickly. Empirically necessary with stop-gradient in BYOL/I-JEPA; theory still debated.</details>

6. Your latent world model has low prediction error. Name two ways it could still be useless.
<details><summary>Answer</summary>(1) Copy-last baseline is just as good: the scene barely changes between frames, so low error is free (always report vs copy-last). (2) Partial collapse: the latent ignores the arm/object (low effective rank or poor probes), so it predicts the static background perfectly. (3) It never saw failures/contacts, so it predicts well only on demo-like data and cannot evaluate rollouts (your G-series finding).</details>

7. Interviewer: "V-JEPA 2.1 vs V-JEPA 2: what changed and why does it matter for robots?"
<details><summary>Answer</summary>A dense predictive loss over all tokens (visible and masked), deep self-supervision at several encoder layers, multi-modal tokenizers for images and video, and scaling. The result is better dense (spatial, temporally consistent) features, which matter for manipulation; they report a sizeable real-robot grasping improvement over V-JEPA 2-AC. Global-only features are not enough for where-to-grasp.</details>

8. How would you test whether a JEPA latent encodes the gripper state?
<details><summary>Answer</summary>Freeze the encoder, collect frames with known gripper opening, fit a linear (then small MLP) probe on train episodes, report R² on held-out episodes against a baseline (probe on a random-init encoder of the same architecture, and on raw downsampled pixels). Pre-register the bar. If the linear probe is weak but MLP strong, the info is there but entangled.</details>
