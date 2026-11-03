# Chapter 12. JEPA world models: what they are and what they are for

## 12.1 What a world model is

A world model predicts the future from the present and an action: f(s_t, a_t) → s_{t+1}. In pixel space that means generating video (expensive, and most pixels are irrelevant to control). In latent space it means: encode o_t to z_t = E(o_t), predict ẑ_{t+1} = P(z_t, a_t), and never decode. The **joint-embedding predictive architecture** (JEPA) is this latent form with a specific training rule: the target z_{t+1} is produced by the *same* encoder (or a slowly-moving copy of it), and the loss is in latent space, ‖P(E(o_t), a_t) − E(o_{t+1})‖.

## 12.2 Collapse, and the three ways to prevent it

The obvious failure: if E maps everything to the same vector, the loss is zero and nothing is learned. Three families of fix, and which your models use:

1. **EMA target encoder** (BYOL, V-JEPA): the target side is an exponential moving average of the online encoder, so the online side chases a slowly moving target it cannot trivially match. V-JEPA 2 uses this with a stop-gradient on the target.
2. **Variance/covariance regularisers** (VICReg): add terms that keep each latent dimension's variance above a floor and decorrelate dimensions, so collapse is explicitly penalised.
3. **SIGReg** (LeWorldModel): a regulariser that pushes the latent distribution toward an isotropic Gaussian via a sketched characteristic-function test, which the LeWM paper argues makes end-to-end training from pixels stable with a single encoder and no EMA. W1 uses this.

What "collapse" looks like in your numbers: the 5-epoch multi-view model had an effective rank of 5.8 out of 192 dimensions; the 100-epoch G0 run had 24–30. Effective rank (the exponential of the entropy of the normalised singular values) is the standard collapse check; G0's protocol required "no collapse" and measured it per epoch (`checks_by_epoch.csv`).

## 12.3 Your two models

**W1** (LeWM recipe): a small ViT encoder from pixels, two views (top and wrist, ContactGuard's multi-view recipe), a pooled 192-d latent per frame, an action-conditioned predictor, SIGReg, 100 epochs on 182 episodes, $3.30. Open-loop error at K = 30: 0.31 vs copy-last 0.89. Reacts to action corruption (timing-shift ratios 1.99, 1.58; episode-swap and frozen-action detected). Arm-pose linear probe R² 0.57. The model behind G10, G11, W1-A/B/C, REAP.

**W2**: frozen DINOv2-S patch tokens (8×8 pooled per view, 384-d each), a learned action-conditioned dynamics head, trained on the Mac, best-val at step 6 000. Arm-pose R² 0.92 (spatial features carry the arm; a pooled vector does not), gripper-open AUROC 0.999. One decision (K = 10 rollouts, H = 30) takes 1.8 s on MPS, so W2 cannot be in the control loop; it is the offline scorer for G7. Lost with a vast instance; W2′ retraining with seed 3073.

The lesson about representations: *spatial beats pooled* for anything about where the arm is; *never PCA-compress* a JEPA latent, because the information sits in low-variance directions (G1 probes).

## 12.4 Three roles, and what the gates said

### 12.4.1 Planner / ranker

Imagine each candidate action chunk; pick the one whose imagined end state is best. The classic latent-planning story (DINO-WM, TD-JEPA, V-JEPA 2's planning results). Tested here as G1b (recorded decoys at other times; top-1 among 8) and its diagnosis, then as the D-JEPA-inspired operator on frozen W1.

Results: W1 top-1 0.60 (bar 0.80); W2 0.55–0.61; the action-only baseline 0.68–0.90 with regret ≈ 0; the ranking operator leaves regret at 0.263; a perfect operator restricted to swapping within the top two would floor at 0.133 with 8 candidates.

Mechanism (chapter 3): on a position-controlled arm the decoys differ mostly in arm state, which the joints know; 182 success-only episodes contain no counterfactual outcomes from which to learn object-state consequences; and with K = 8 fixed candidates, even perfect ranking is bounded by the candidate set's quality (the 0.133 ceiling is a property of the candidates, not the ranker). The Planning Limits paper adds the horizon point: rollouts shorter than the target distance rank below chance even with a perfect simulator (G1b's H = 5 at d = 30 scored 0.058).

What would change the verdict: object-state differences (G7), executed-outcome labels for every candidate (D-JEPA's actual supervision; a robot day with resets), and a harder task where the arm's path does not determine the outcome.

### 12.4.2 Progress and stall signal

Latent distance to the goal should fall as the task progresses; a stall is when it stops falling. No labels. Tested as G6a (W1, W2) and G6b (H-JEPA's level-2 slow features with SIGReg, stride 15).

Results: monotonicity Spearman −0.32 to −0.50 (bar −0.8); G6b −0.73; false stall rates 32 % and 52 %. Mechanism: manipulation trajectories loop in physical space (approach, lift, swing, place), so distance to the final frame rises and falls on the way; coarsening time does not remove the loop. The literature agreed in advance (TD-JEPA: 0.65 raw distance vs 0.95 learned cost on their benchmark). A progress signal has to be trained on progress labels; nobody has them for free.

### 12.4.3 Verifier and monitor

Read the outcome from the latent at rest; flag surprise (prediction error of the next latent) during execution. Tested as G10 (retrospective), G11 (latency), W1-A, G11-B-cal, W1-C1, W1-C2 (fresh rollouts), and REAP.

Results: at the judge timestamp, W1's latent reads the G9c-3 verdict at AUROC 1.000 (W2 0.976); 30 steps earlier the imagined latent beats policy identity by +0.18; surprise spikes 1.82× on missed grasps and 2.58× on stalls. Latency fits (17.6 ms per check). But on fresh rollouts: the live thresholds fire on 42/42 successes; a refit catches 3/25 failures at 1/44 false alarms; the imagined latent is *worse* than the current one for an event-anchored grasp check (0.70 vs 0.82); the conformal band as registered is not conformal across time.

Mechanism: the readout works because the outcome *is* visible in the final frame (nested or not), and a latent that encodes the scene encodes it. Anticipation on seen data worked because thresholds were fit on the same distribution; on fresh data the surprise distribution is wider (USB timing jitter; different layouts) and the signal-to-noise for "failure in 1 s" is low at the frame level. Imagination being worse than the current latent for a grasp check says the predictor adds noise to a signal the encoder already has.

Decision: shadow mode. The world model scores every trial and the scores are logged next to the verdict; it never stops the arm. It becomes a veto only if a fresh-rollout gate passes against the joint baseline, and the candidate gate is object state (G7).

## 12.5 What the field's recent papers add

- **D-JEPA** (2609.24749): prediction accuracy does not make latent distance decision-aligned; train a separate operator on executed outcomes. Your proxy-label version is not that test.
- **One-step fidelity is a poor proxy** (2610.19375): across LeWM checkpoints, one-step error and planning success correlate at r ≈ −0.01; a grounding head is indistinguishable from more epochs. So checkpoint selection by loss is meaningless for control, and §7 needs its matched control.
- **RoboJEPA** (2610.10515): scaling latent world models on V-JEPA 2.1-G features; action conditioning works on your arm (9–10 % below shuffled) but the encoder is 526 ms on the Orin. The scaling angle (§5) is the GPU-poor end of their curve.
- **H-JEPA** (2610.06805): hierarchy across timescales; tested as G6b, failed on loops.
- **Physical-state grounding** (2608.06799, 2609.03565): auxiliary losses to joint state; §7, with the 2610.19375 control.
- **CF-JEPA** (2610.00727): controllable/background factorisation; the lighting angle (§8).
- **LeWAM ×2, JEPA-WAM, PLaW-VLA, WAVE-0**: end-to-end world-and-action models; compute-heavy; reading, not testing.

## 12.6 The thesis, stated carefully

On a cheap position-controlled arm with a few hundred success-only episodes, a JEPA world model trained from pixels (a) learns real action-conditioned dynamics (G0), (b) reads task outcome from the at-rest latent perfectly (G10 step 1), (c) does not rank action chunks better than the commanded joints (G1b, operator), (d) does not provide a progress signal from latent distance (G6), and (e) does not provide a live failure monitor that meets the bar on fresh rollouts (W1-A/C; REAP v2 fail at AUROC 0.79 with the confounds removed, though the head is action-conditioned). The open question is (f): whether it reads object-state consequences the joints cannot (G7). The honest role today is verifier-at-rest and logged signal.

Every clause has a gate ID. That is the sentence to say in an interview, and the gate IDs are what make it a research result rather than an opinion.

## 12.7 Exercises

1. Derive why an EMA target prevents collapse (what happens to the loss gradient if the online encoder tries to collapse while the target lags). Then write SIGReg's objective in one line from the LeWM paper and say what distribution it pushes the latent toward.
2. Compute effective rank for W1's latents on 1 000 frames; compare to the 24–30 reported. Then on the 5-epoch checkpoint if it exists.
3. Reproduce G10 step 1 (AUROC 1.000) with a logistic probe on W1's at-rest latents and the G9c-3 labels; then do it on *fresh* episodes the probe was not fit on (5-fold by episode). Report both. Write two sentences on what the gap means.
4. Write the G7 decision sentence (chapter 1) and the three outcomes (pass, fail, not proven) with what each does to the thesis in 12.6.
