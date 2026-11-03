# Chapter 3. The baseline is the method

## 3.1 The principle

Compute the cheapest thing that could explain the result **before** training or evaluating the method. If the method does not beat it, stop.

This is the step that, in this repository, retracted more claims than any other. It is also the step most papers skip, which makes it the first thing your test adds to the literature. A method that beats a strong published competitor but not a trivial one has not been shown to do what its paper says; it has been shown to exploit something in the data that the trivial one also exploits.

## 3.2 Why: shortcuts are the default

Learned systems find the cheapest fit. If a trivial signal predicts the target, the model learns the trivial signal, and a paper that does not report the trivial signal cannot tell you whether its method learned anything else. The baseline is the null hypothesis made concrete. It is not "the method is useless"; it is "the method is indistinguishable from this one-line computation on this data", which is a precise and actionable statement.

The baselines are cheap for a reason: they encode what you already know about the hardware and the data. On a position-controlled arm, the actions are the future arm state. In teleop data, failures are longer. In mixed-policy rollouts, the weak policy makes the failures. Each of these is a fact about the setup that a learned method must beat to earn its place.

## 3.3 The six baselines, with the derivation of each

### 3.3.1 Commanded joints

On a position-controlled arm, the controller drives each servo to the commanded angle. Barring collision, saturation or a large lag, the arm *will be* where the commands say, with a 2–4 frame lag you measured. So a chunk's final commanded joint vector, passed through forward kinematics, is a near-perfect prediction of the arm's end state, with no images and no model.

What this means for latent world models: anything they predict about the *arm* is already known. Their only possible value is in what the actions do not determine: object state (did the bowl come along), contact (did the gripper close on the rim or on air), and failure (did something unexpected happen). G1b asked the world model to rank chunks whose differences were mostly arm-state differences (recorded decoys at other times), so it was competing with the joints on the joints' home ground, and lost (action-only top-1 0.68–0.90, regret ≈ 0; W2 regret 0.18, W1 0.26). The diagnosis note in `G1_PROTOCOL.md` is the clearest statement of this in the repo: "G1b measured kinematics, not world-model value."

The consequence is G7: test the world model on object-state differences, where the joints are blind. That gate has been waiting for the data to exist since 10-13.

### 3.3.2 Episode length

Failures take longer (stalls, retries, dithering) or are cut short (timeouts, operator stops). So episode length predicts outcome, often very well. On ArmnetBench, length alone scored 0.934–0.938 AUROC; Robometer-4B scored 0.861 on the partial set and 0.818 on the complete one. On the G9b re-analysis, length beat Robometer on all four held-out tasks (0.86–0.97 vs 0.69 median). A judge that does not beat length may simply have learned length from the video.

How to control for it: report AUROC "beyond length": fit the baseline on length alone, then measure the gain from adding the judge score (ArmnetBench's protocol uses quintile medians and cross-validated gain; the G9b re-analysis found +0.008 with a CI including 0). Or stratify: compare judge scores within length quintiles.

The counter-case: a judge that *should* beat length and does not is informative about the judge; a judge that beats length but only because the failure mode in your data is length-correlated (stalls) may fail on a dataset where failures are fast (wrong object). G9a-own's note "wrong-object failures score highest" is this: Robometer's scores were anti-correlated with outcome on the failure type that was not length-correlated.

### 3.3.3 Policy identity

When several policies produced the rollouts and one of them made most of the failures, "which policy ran" predicts outcome. C1 (2026-10-11): policy identity 0.76 vs the best latent probe 0.72, because TurboVLA made most of the 15 failures. REAP (2026-10-23): the risk head's 23/25 on the 69-rollout set looked strong until the count showed 17 of 25 failures were TurboVLA, which fails visibly early; on the 8 non-TurboVLA failures it caught 6 at 7/42 false interventions.

Control: report the single-policy subset (or the subset excluding the dominant failing policy) as the primary result, the mixed set as secondary, never pooled. The REAP README now does this. Any monitor or judge evaluated on mixed-policy rollouts without this split is suspect.

### 3.3.4 Copy-last

For a dynamics model: predict ẑ_{t+1} = z_t. If the world is slow relative to the frame rate, copy-last is excellent at short horizons and the only thing that can beat it is actually modelling the dynamics. G0's bar was open-loop error at K = 30 relative to copy-last (0.31 vs 0.89: pass). The corruption checks are the same idea applied to action conditioning: shuffle or time-shift the actions and the error must rise (ratios 1.99 and 1.58 at ±0.5 s). A world model that beats copy-last at K = 1 and not at K = 30 has learned to copy.

### 3.3.5 The mean of N samples

For best-of-N selection: average the N candidates. If the samples are bundled around a shared bias, their mean is as good as any selection and costs no model. H0: the 8 VLA-JEPA chunks had a 1.08° spread; the mean captured 45 % of the oracle gain; reranking added 1.84× latency to capture the rest at best. The baseline killed the reranker in one afternoon.

### 3.3.6 The current component

For a replacement: the thing already in the stack. A new judge is compared to G9c-3 (0 FP, 94.3 %). A new chunk-boundary method is compared to prefix overwrite (−72 %, +0.4 ms). A new monitor is compared to "no monitor" (0/42 false interventions, 0/8 caught). The current component is often the strongest baseline because it was itself selected against the trivial ones.

## 3.4 The question to ask of every paper

*What trivial thing did they not compare against?*

Read the baseline table of any robotics paper with the six above in mind. Judges without episode length. Latent planners without commanded joints. Rerankers without the sample mean. Monitors without "elapsed time". Curation metrics without episode length (the smoothness audit in the prep queue exists because "rinse" reported smoothness without controlling length). The missing baseline is the first thing your reproduction adds, and it is usually the one that matters.

## 3.5 Ordering: baseline before model

The protocol template in the prep queue has "Baselines (computed first)" as a section before "Method under test", and the Claude Code instruction says to compute and report them before any training. The ordering is not bureaucratic. A baseline computed *after* the model's number is known gets rationalised ("length is 0.93 but that's an artefact"); a baseline computed first sets the bar the model must clear, and if it is already at 0.93 you may decide not to train at all. Three of the eleven prep-queue bars are "beat episode length by ≥ 0.10 with the paired CI excluding 0"; those are decisions the baseline makes cheap.

## 3.6 When the baseline wins: what to write

A baseline win is a result. Write it as one:

- The baseline's number, with CI, and the method's, with the paired CI of the difference.
- What the baseline encodes about the data (length: failures stall; joints: the arm is position-controlled).
- What the method would need to beat it (object-state differences; failures that are not length-correlated; a task that is not at ceiling).
- Whether that need can be met on this hardware, and if so, which gate tests it.

The G1b diagnosis is the model: it names the baseline, derives why it wins (kinematics), states what the world model could still add (object state, contact), and points to the gate (G7). That is a research contribution, not a failure report.

## 3.7 What the interviewer will ask

*"How do you know your world model learned anything?"* G0's copy-last and corruption ratios. Then: "and G1b showed that what it learned about the arm was already in the joints, which is why we test it on object state now."

*"Your judge gets 0.94 AUROC. Is that good?"* "Episode length gets 0.93 on the same rollouts, so 0.94 alone is not; the gain beyond length is what we report, and for Robometer it was within noise of zero. For our deterministic judge the relevant number is zero false positives, which length cannot give."

*"What baseline do you wish the papers you read would report?"* Pick one and say why: episode length for every judge and monitor paper; commanded-joint rollouts for every latent-planning paper on position-controlled arms.

## 3.8 Exercises

1. For each of the six baselines, write the one-line computation (pseudocode) and the data it needs. Run episode length and policy identity on the 69 rollouts yourself; reproduce 0.88 for policy identity on the G9a-own set.
2. Take three papers from the prep queue. For each, write the baseline they did not run and estimate from their own tables what it would have scored.
3. Build the kinematic chunk scorer (3.3.1) from the SO-101 URDF in `shared/robot/` and the measured lag; reproduce the action-only top-1 range on the G1b anchors.
