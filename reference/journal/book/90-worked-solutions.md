# Appendix S. Worked solutions

Solutions to the exercises of chapters P1, P2, 3, 7, 8 and 10a. Exercises that ask you to run something on your own data (most of chapters 3 and 8) have a **method** and the **expected result** where the register already holds it; the point is to reproduce it, not to read it. Every computed number below was recomputed on 2026-10-24 (SciPy for exact intervals). Try each exercise before reading its solution: a solution read first teaches recognition, not recall (chapter F.1).

Doing these exercises found errors in the book, now corrected in place: the Hanley–McNeil standard errors in 7.5 (0.045 → 0.053 and 0.08 → 0.089), the 180/200 Wilson upper bound in 7.4 (0.94 → 0.93), the 60-vs-90 % sample size in 7.4 (35 → 32), and the AUROC 0.93 standard error in 7.7 (0.08 → 0.064). That is what exercises are for.

## S.P1 Chapter P1, deep learning from zero

**1. Two steps of gradient descent by hand.** Data (1, 3), (2, 5), (3, 7); model ŷ = wx + b; start w = b = 0; η = 0.05.

- Step 0: predictions 0, 0, 0; errors −3, −5, −7; loss (9 + 25 + 49)/3 = 27.667. ∂L/∂w = 2·mean(−3·1, −5·2, −7·3) = 2·(−34/3) = −22.667; ∂L/∂b = 2·mean(−3, −5, −7) = −10. Update: w = 0 + 0.05·22.667 = 1.133; b = 0 + 0.05·10 = 0.500.
- Step 1: predictions 1.633, 2.767, 3.900; errors −1.367, −2.233, −3.100; loss (1.868 + 4.988 + 9.610)/3 = 5.489. ∂L/∂w = 2·mean(−1.367, −4.467, −9.300) = −10.089; ∂L/∂b = 2·mean(−1.367, −2.233, −3.100) = −4.467. Update: w = 1.133 + 0.504 = 1.638; b = 0.500 + 0.223 = 0.723.

These match the table in P1.2. The plain-Python loop is twelve lines: two lists, a loop over steps, the two gradient formulas, the two updates. With η = 0.3 the second step already has a loss of 150: the step overshoots the minimum by more than it started from, and every step makes it worse.

**2. Backward pass with w = (1, 1), x = (2, −1).** s = 1·2 + 1·(−1) = 1; e = s − 1 = 0; L = 0. ∂L/∂e = 2e = 0, so every gradient is 0: ∂L/∂w₁ = ∂L/∂w₂ = 0. The lesson: at a perfect fit, gradient descent stops moving. The gradient measures how much the loss would change, and at a minimum it would not.

**3. Softmax and cross-entropy of logits (0, 0, 0).** All exponentials are 1, so each probability is 1/3. Cross-entropy is −ln(1/3) = ln 3 = **1.099** whatever the correct class. A model that knows nothing assigns equal probability to every class, and its loss is ln(number of classes). This is the "initial loss ≈ ln V" check of P2.20.

**4. Attention output for token 3.** Its query (1, 1) gives scores (1, 1, 2)/√2 = (0.707, 0.707, 1.414) against the three keys; softmax gives (0.248, 0.248, 0.504); the output is 0.248·(10, 0) + 0.248·(0, 10) + 0.504·(5, 5) = **(5.00, 5.00)**. Exactly the middle because token 3's query is symmetric between the first two keys: it gives them equal weight, their values (10, 0) and (0, 10) average to (5, 5), and its own value is (5, 5) too.

**5. Collapse in three sentences.** A JEPA is trained to make its prediction of the next frame's summary match the summary the encoder produces for that frame. If the encoder learns to produce the same summary for every image, the prediction is trivially perfect and the loss is zero, but the summaries say nothing about the images. The defence W1 uses is SIGReg, which penalises summaries that do not spread out like a standard normal distribution along random directions, so a constant output is heavily penalised.

**6. Why MSE fails on left/right demonstrations, and flow matching does not.** MSE regression predicts the average of the demonstrated actions for a given observation, and the average of "left" and "right" is "straight ahead", into the obstacle, an action no demonstrator took. Flow matching learns to transport random noise to the demonstrated actions; different noise samples land on different modes, so each executed action is either left or right, never their average.

**7. AUROC for positives (0.8, 0.6) and negatives (0.7, 0.2, 0.1).** Six positive–negative pairs: 0.8 beats all three (3); 0.6 beats 0.2 and 0.1 but not 0.7 (2). AUROC = 5/6 = **0.83**.

## S.P2 Chapter P2, PyTorch for the coding round

**1–2.** Run the blocks; there is nothing to look up. The split-heads shapes: (2, 5, 32) → `view` → (2, 5, 4, 8) → `transpose(1, 2)` → (2, 4, 5, 8) and non-contiguous → back with `transpose(1, 2).contiguous().view(2, 5, 32)`.

**3. The three linear-layer gradients.** With y_ni = Σ_j x_nj W_ij + b_i and upstream gradient g_ni = ∂L/∂y_ni:

- ∂L/∂W_ij = Σ_n g_ni · x_nj, so ∂L/∂W = gᵀx, shape (out, in).
- ∂L/∂b_i = Σ_n g_ni, so ∂L/∂b = g summed over the batch, shape (out,).
- ∂L/∂x_nj = Σ_i g_ni · W_ij, so ∂L/∂x = gW, shape (N, in).

Check: shapes must match the tensor each gradient belongs to. That is the fastest way to catch a missing transpose.

**4. The five explanations.**

- *`detach` on the JEPA target:* the target is treated as a fixed answer, so the encoder is not also trained to move the target towards the prediction, which is one route to collapse.
- *`eval()` vs `no_grad()`:* `eval()` changes how layers behave (dropout off, batch norm uses stored statistics); `no_grad()` stops recording the graph for gradients. Evaluation needs both; neither implies the other.
- *Why bf16 needs no loss scaler:* bf16 has the same exponent range as fp32, so small gradients do not underflow to zero; fp16's narrow range does, which the scaler compensates.
- *The mask conventions:* `nn.MultiheadAttention` takes a mask of positions to *block* (True = not allowed); `F.scaled_dot_product_attention` takes positions to *allow* (True = allowed). Same shape, opposite meaning; passing one to the other silently attends to exactly the wrong positions.
- *What breaks going 1 → 8 GPUs:* the effective batch is 8× larger, so the learning rate and warm-up usually need changing; the sampler must be distributed and its epoch set each epoch; logging, evaluation and saving must happen on rank 0 only; saved keys get a `module.` prefix unless you save the unwrapped model; results stop being bit-for-bit reproducible.

## S.3 Chapter 3, the baseline is the method

**1. The six baselines as one-line computations.**

| Baseline | Computation | Data needed |
|---|---|---|
| Commanded joints | FK(final commanded joint vector of the chunk) as the predicted end state; rank candidates by its distance to the goal | the chunk's actions; the URDF |
| Episode length | score = number of frames; AUROC of length against the outcome | episode lengths; outcome labels |
| Policy identity | score = the failure rate of the policy that produced the episode, estimated on the other episodes (leave-one-out) | the policy name per episode; outcome labels |
| Copy-last | predicted next latent or frame = the current one | the sequence itself |
| Mean of N samples | execute the average of the N candidate chunks | the N chunks |
| Current component | the same probe or judge on the *current* encoding instead of the imagined one | the encoder |

Expected result for the reproduction: policy identity reaches **0.88** AUROC on the G9a-own set (the register's number). Compute it leave-one-out; a version that uses each episode's own outcome to estimate its policy's failure rate leaks the label and scores higher than it should.

**2. Baselines the papers did not run.** The method is chapter 2a.3: for each paper, look for the cheapest row that could produce its headline number (length, the commanded action, copy-last, random at the same budget, the frozen base model with the same extra training steps). Estimate it from their own tables where possible, for example a curation paper's random-at-the-same-budget row if it exists, or the "w/o" ablation closest to "nothing". The answer is a short table per paper, not a number from this appendix.

**3. The kinematic chunk scorer.** Method: for each G1b anchor and each candidate chunk, take the final commanded joint vector; apply the measured lag (2–4 frames) by using the command that many frames before the end; compute the gripper position with the URDF FK of chapter 10a.2; score each candidate by the distance of that position to the goal frame's FK position (from its recorded state); top-1 = whether the true chunk ranks first. Expected: **top-1 0.68–0.90** across the anchor sets, with regret ≈ 0, against the world model's 0.60. If your number is far below, check the lag direction and that commanded and recorded joints use the same calibration.

## S.7 Chapter 7, uncertainty

**1. Wilson and Clopper–Pearson** (95 %):

| k/n | Wilson | Clopper–Pearson |
|---|---|---|
| 9/10 | [0.60, 0.98] | [0.55, 1.00] |
| 8/10 | [0.49, 0.94] | [0.44, 0.97] |
| 10/10 | [0.72, 1.00] | [0.69, 1.00] |
| 6/6 | [0.61, 1.00] | [0.54, 1.00] |
| 2/6 | [0.10, 0.70] | [0.04, 0.78] |
| 18/20 | [0.70, 0.97] | [0.68, 0.99] |
| 54/60 | [0.80, 0.95] | [0.79, 0.96] |
| 180/200 | [0.85, 0.93] | [0.85, 0.94] |
| **23/25** | **[0.75, 0.98]** | **[0.74, 0.99]** |
| **4/44** | **[0.04, 0.21]** | **[0.03, 0.22]** |

Clopper–Pearson is always at least as wide (it is exact and conservative). Implementation: Wilson is the closed form of 7.4; Clopper–Pearson is `scipy.stats.beta.ppf(α/2, k, n−k+1)` for the lower bound and `beta.ppf(1−α/2, k+1, n−k)` for the upper, with 0 and 1 at the edges.

**2. Paired episode bootstrap on C1.** Method: for each episode, the three scores (imagined probe, current probe, policy identity) and the outcome. Draw R = 10 000 resamples of episode indices with replacement (the same indices for all three scores); compute the three AUROCs on each resample; report each pairwise difference's 2.5th and 97.5th percentiles. Expected direction from the register: current probe 0.82 above imagined 0.70; policy identity 0.76 against the best probe 0.72 in the earlier C1 run. A difference whose interval includes 0 is *not proven*.

**3. What 207 trials can resolve.** At α = 0.05 and 80 % power, two arms need about **32 trials each** to separate 60 % from 90 % (a 30-point gap), and about **62 each** to separate 70 % from 90 % (a 20-point gap). So 207 trials allow about three two-arm comparisons at the 30-point level (3 × 64 = 192) or one at the 20-point level (124) with room for a smaller one. A cut to 150 trials that loses nothing you would act on: keep the comparisons whose outcome would change a decision (each at 32 per arm, so at most two full comparisons plus a pilot), and drop any arm whose expected gap is under 30 points, because 150 trials cannot resolve it anyway and the result would be *not proven* whatever happened.

**4. Hanley–McNeil standard error of an AUROC.** With A the AUROC, n₊ positives and n₋ negatives: Q₁ = A/(2 − A), Q₂ = 2A²/(1 + A), and

SE² = [A(1 − A) + (n₊ − 1)(Q₁ − A²) + (n₋ − 1)(Q₂ − A²)] / (n₊ · n₋).

At A = 0.85: A² = 0.7225; A(1 − A) = 0.1275; Q₁ = 0.85/1.15 = 0.7391, so Q₁ − A² = 0.0166; Q₂ = 1.445/1.85 = 0.7811, so Q₂ − A² = 0.0586.

- (n₊, n₋) = (8, 42): numerator = 0.1275 + 7·0.0166 + 41·0.0586 = 0.1275 + 0.1163 + 2.4021 = 2.6459; divided by 336 = 0.007875; **SE = 0.089**.
- (25, 44): numerator = 0.1275 + 24·0.0166 + 43·0.0586 = 0.1275 + 0.3989 + 2.5193 = 3.0457; divided by 1 100 = 0.002769; **SE = 0.053**.

The smaller class dominates: with 8 positives, the standard error is about 0.09, so two AUROCs 0.10 apart on that set cannot be ranked. (At A = 0.93 on (8, 42) the SE is 0.064: high AUROCs have smaller standard errors.)

## S.8 Chapter 8, reading a failure

**1. Task-swap probe on the 21-task SmolVLA.** Method: pick three instruction pairs that differ only in the target ("…blue bowl…" vs "…red bowl…"); for a set of frames from validation episodes, run the policy twice per frame with the two instructions and the same images, state and noise; report the per-frame difference of the first action (or of the chunk mean) in degrees per joint, as a distribution, not just a mean. Then repeat with the images zeroed. Expected: a policy that reads its instruction moves several degrees on the frames where the target matters (SmolVLA moved 7–13° in the TurboVLA comparison, chapter 8.3) and almost nothing on frames where the arm's pose already decides; with images zeroed, a large remaining difference means the policy reads text without vision, and a collapse to near zero means instruction-following depends on seeing the scene.

**2. The 8.6 mechanism form for G6a, W1-A and G16-B.** Method: one page each, in the six-line form of 8.6 (observation, probes and their results, the stage, the mechanism, the generalisation, what it changed), using the register rows and reports. The test of a good page: someone else can predict a new result from your mechanism line.

**3. Path A, before and after.** There is no single answer; the exercise is honesty about prediction. The true paragraph: the faithful upstream implementation, trained on the same data, moved 0.020° under a task swap against a 0.105° control, so the instruction is ignored by the architecture as the paper trained it on this data as well, which locates the cause in the data (fact 7: the instruction is redundant on 94 % of frames) and withdraws the claim that the port was at fault. Grade your two earlier predictions by which one named the data as the likely cause.

## S.10a Chapter 10a, robotics fundamentals

**1. Two-link arm with lengths 0.10 and 0.10 m.** The reachable set is every point within 0.20 m of the base (the inner radius |l₁ − l₂| is 0, so even the base point is reachable, with the elbow folded flat). The original test point (0.140, 0.195) is 0.240 m away: **no solution**. Any target between 0.20 and 0.26 m that was reachable with 0.12/0.14 m links loses its solutions.

**2. IK with orientation.** Method: from the FK transform take the rotation R; define an orientation error as the rotation vector of R_target · R_currentᵀ (or, simpler, the differences of two of the gripper's axes); stack it under the position error to get a 6-element error and a 6 × 5 Jacobian (numerical, as in 10a.2); solve with the same damped least squares. Expected: with five joints and six constraints the problem is now **over-constrained** in general, so the solver finds the best compromise; for a target generated by FK from a real configuration, it should return that configuration (up to joint-limit clipping), removing the redundancy seen in 10a.2.

**3. Gains and delay.** With Kd = 0.2 in the simulation of 10a.4, a 30 ms measurement delay leaves a wide stable range: the largest Kp that still settles within 1 s is about **3.9** (larger gains oscillate longer; Kp = 5 did not settle in 2 s). With a **240 ms** delay no Kp in 0.1–10 settles within 1 s, and every Kp tried from 0.1 to 1.0 became unstable: the damping term, acting on a quarter-second-old velocity, pushes at the wrong time. The paragraph this asks for: a learned policy that outputs goals at about 4 Hz must not be the inner feedback loop of a joint. The fast loop stays in the servo, which runs on fresh measurements; the policy only moves the servo's target, and the chunk boundary is where its delay shows (10a.5).

**4. Camera intrinsics.** Result depends on your camera. A C920 at 640 × 480 typically calibrates to a focal length of several hundred pixels and a reprojection error well under one pixel; if the near-versus-far pixels-per-centimetre ratio on the mat is far from the model's prediction (about 2 : 1 for the geometry in 10a.7), recheck the camera's height and angle before trusting either number.
