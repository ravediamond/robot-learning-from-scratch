# Chapter 13. Judges, monitors and data

Three topics that share one fact: on your hardware, the limiting resource for learning is labelled outcome, and the limiting resource for labelled outcome is a judge you trust.

## 13.1 Judges: what they are for

A judge maps a finished rollout to success / failure (binary) or to a progress score. It is needed for: (a) evaluating policies without an operator at every trial; (b) selecting which rollouts enter the next training set (the data flywheel); (c) reward for RL post-training. Each use has a different cost of error. For (b), a false positive (failure called success) poisons the training set and is the error to drive to zero; for (a), both errors cost equally; for (c), calibration matters as well as rank.

### 13.1.1 The four tiers (v1.5 §1.1)

1. **Physical measurement**: a scale (the LEGO cell: brick weight), a count, a limit switch. Cheapest, most trusted, task-specific.
2. **Deterministic perception at rest**: G9c-3. HSV masks for the two bowls on the top camera, centroid geometry, "is blue inside pink", evaluated after 0.5 s of stillness or timeout. 94.3 % accuracy, 0 false positives, every failure type caught; the 12 errors are successes whose recording continued past the operator's reset.
3. **Learned or VLM judge**: Robometer, Flora, Qwen3.5, Gemini. General; must earn its place per task through a report card (FP, FN, ambiguous rate against human labels).
4. **Human label**: ground truth; slow; the reference for everything above.

Rules: tiers 1–2 may decide alone; tier 3 pairs with a tier-2 verdict and supplies explanation; tier-3 verdicts enter `demos/` only with an human spot-check at a frozen rate; a tier-3 judge that passed on one policy's failures is re-checked when the policy improves (coverage shift, 2610.24801: false-accept rate on remaining failures rose from 31 % to 50 % as a policy went from 67 % to 97 % success).

### 13.1.2 What the gates taught about learned judges

- **Zero-shot reward models do not transfer** to this arm. Robometer-4B: 0.58–0.62 on your rollouts; 0.818 on ArmnetBench (length 0.938); on four held-out ArmnetBench tasks, length beats it on all four (median 0.69 vs 0.86–0.97); adding it to length gains +0.008 with a CI including 0. The failure mode that matters on bowls (wrong object) is an identity question; a trajectory-comparison model has no reason to answer it, and in fact scored wrong-object failures *highest*.
- **VLM judges fail mid-episode and pass at the end with the whole episode.** Sliding-window FAIL probability (Gemini Flash): threshold fit at 0.98, 0/8 failures caught, ~10 s per call. Two checkpoint calls (after grasp, at rest): 41/42 successes flagged, because the post-grasp checkpoint fires before the lift and nested bowls read as "blue alone" from the top. Whole-episode input (64 frames, top and wrist side by side, one terminal question): Flora 8/8 at 2/42 (AUROC 1.00, deterministic, 6 s); Qwen3.5-27B 7/8 at 2/42 (0.91); Qwen3.5-9B 16/42 successes flagged; Gemini Flash 8/8 at 3/42 (0.96). The format and the size did the work; the robot fine-tune added a little.
- **Episode length is the baseline every judge must beat**, and the one most papers do not report.

### 13.1.3 Building a tier-2 judge for a new task

The recipe from G9c, generalised: identify the one visual relation that defines success at rest (nesting; a brick in a slot; a count); find the camera and the colour/shape cue that makes it separable; write the deterministic rule; define the end rule (stillness-or-timeout, because failing policies rarely return to rest: G9c-2 marked 29/88 invalid when it required rest); measure FP and FN against human labels on ≥ 50 episodes; set the decision threshold for FP = 0 and report the FN it costs. RAFAIL and Robot-WATCHDOG (prep queue) are the learned generalisation of this: object tracks → relations → classifier; the test is whether they match G9c-3's zero FP and transfer to cubes.

## 13.2 Monitors: predicting failure before it finishes

A different problem: act *during* the episode, with a budget of false interventions. The cost structure: a missed failure wastes the rest of the episode and may damage something; a false intervention wastes a good trial and erodes trust. Your bar: ≥ 5/8 failures caught at ≤ 1/20 interventions on successes, on held-out folds.

Signals tried: W1 prediction error (surprise), cosine drift of the latent, a logistic or MLP risk head on [W1 latent, commanded actions, step] (ReAP-style), elapsed time, episode length, conformal bands on the surprise trace. Results on 50 fresh rollouts, v1: nothing clears the bar; the logistic head is closest (6/8 at 7/42, AUROC 0.93); W1 surprise 2/8 at 2/42; drift nothing. Two confounds: the secondary set's failures are 17/25 TurboVLA; the onsets are Gemini-labelled with two 110- and 289-frame misses.

**v2 (2026-10-23)** removed both with ReAP's and FARM's own devices: ramp target (no onsets), no time feature, common-horizon truncation, shuffled/zero-action controls, bTWA, VLA-Corrector's GRU as floor. Logistic **3/8 at 5/42, AUROC 0.789**; MLP 6/8 at 13/42; GRU 2/8 at 3/42. Verdict **fail**. The head is action-conditioned (+0.08 over shuffled) but weak; the time feature and episode length had produced the 0.93. The next version changes what W1 is trained on (the deployed policy's own rollouts, ContactGuard's recipe), not the head. The 25 hand-labelled onsets are now a timing check, not a blocker.

Latency budget on the Orin: 17.6 ms per W1 check (fits the 33 ms tick); 153 ms per 25-step imagination (off the control thread); the policy's chunks slow from 238 to 304 ms while a monitor shares the GPU. Any monitor design must state where it runs and what it costs the policy.

The P² paper (prep queue) proposes a cheaper signal than all of these: a linear probe on the *policy's own* activations, trained on successes only, reading task progress; a run that falls behind expected progress is flagged. It adds ~0 ms because the activations already exist. It is the first thing to test after the onsets are clean.

## 13.3 Data: the scarce resource

### 13.3.1 What a demonstration contains and what it teaches

Per frame: two images, six follower joint angles (`observation.state`), six leader joint angles (`action`), a timestamp, a task string. Per episode: 15–25 s at 30 fps, one reset, one operator. What the policy learns from it is whatever reduces loss (chapter 11), so the properties of the data decide what gets learned:

- Idle frames at the start → "stay" is a valid action → the policy freezes (TurboVLA until trimming). Trim every dataset with the same script.
- Two instructions on near-identical frames → the instruction is redundant with the state on 94 % of frames → copycat (fact 7). Instruction-conditioned behaviour needs data where the instruction changes the target early and often.
- No corrections in the data → no recovery behaviour → oscillation at the first off-distribution state. The tidy-loop demos with deliberate corrections are the fix.
- Failures in the training set → the policy learns failures. Rule 1.2 in AGENTS.md: failures train the world model, the cost and the judge, never the policy. One key press per episode, right after it, is the label discipline.

### 13.3.2 Curation signals, and which to trust

A typical robot-data research posting asks "which data improves robot learning, how to make it useful, when to introduce it". The signals you can compute, and what is known:

- **Outcome** (success/failure from a judge): the only signal with a mechanism; needs a judge you trust (13.1).
- **Episode length**: predicts outcome; as a curation signal it selects for fast demos, which may be skilled or may be lucky.
- **Trajectory smoothness** (SAL etc., prep queue 1.1): the audit paper's claim is that it ranks operator skill, not success, once length is controlled. Your partial-correlation test will say.
- **State criticality** (prep queue 1.2): how much a small action error at a state changes the outcome; should recover the 6 % decision band if it works; would let you weight the frames that matter.
- **Quality-aware weighting vs filtering vs conditioning** (prep queue 3.2): the ablation design, with your signals in place of simulator labels.
- **Layout coverage** (v1.5 §2): failed layouts first, then farthest-point coverage of the workspace; the active-selection angle (§11) tests it against random.

The habit: any curation signal is tested as a judge first (does it predict outcome beyond length?) and as a training weight second (does the reweighted fine-tune win on a gate with enough n?). The second test needs n ≈ 126 (pooled over 21 tasks) to see a 15-point gap, which is why the prep-queue bars for 3.1 and 3.2 are written on pooled success.

### 13.3.3 Augmentation

RESTA (prep queue 3.1): reverse demonstrations for reversible primitives (your forward/undo pairs make half the tidy-loop primitives reversible) and resample stride. Free data if the reversal is physically valid; the protocol must list which primitives qualify. Any augmentation is tested with the same pooled-gate rule.

### 13.3.4 Provenance and the Hub

Datasets are versioned by commit on the Hub (the 69 rollouts at `bad1c8e`); normalisation statistics are per dataset; checkpoints that protocols depend on go to the Hub the day they exist (W2 was lost; lesson 24); the Jetson disk is not storage. Every register row names the dataset commit it ran on.

### 13.3.5 What a robot day produces

~207 trials in 8 h 15 (the runbook's estimate; chapter 7 says plan for 3 two-arm comparisons at the 30-point level). Each trial: a dataset episode (recorded through the harness), a label (key press), a judge verdict at rest, a layout id, an executor version, timing. That record is what the v1.5 flywheel consumes, and it is the output that makes a robot day worth more than its trials: the trials are evidence for today's gates; the records are training data for next month's.

## 13.4 What the interviewer will ask

*"How do you evaluate a policy without a human watching?"* The tiers. "A deterministic judge with zero false positives on the task I have; a VLM judge with whole-episode input as a second opinion with a report card; human labels on every gate trial."

*"Why not just use a reward model?"* Robometer's numbers, length as the baseline, the wrong-object failure mode. "They answer 'how far along is this', not 'did it pick the right object', and on my tasks the second is the failure."

*"What data would you collect next, and why?"* Failed layouts first (v1.5), with the test that makes it a claim rather than a policy (§11, selected vs random, pooled gate). And: policy rollouts (successes and failures) for the world model's training set, because every monitor result so far was on a model that never saw this policy fail.

*"How do you know your curation signal isn't just episode length?"* Partial correlation controlling length; the smoothness audit; the G9b "+0.008, CI includes 0" line.

## 13.5 Exercises

1. Implement the G9c-3 judge from its config on 10 rollouts by hand (masks, centroids, the inside test, the end rule). Reproduce the verdicts.
2. Run the whole-episode Qwen3.5 judge format on 5 of your cube episodes (no labels yet): write what it says and what you would need to score it.
3. Compute episode length, smoothness (SAL) and outcome on the 69 rollouts; the raw and the partial correlations. Decide, before the prep-queue run, whether you expect smoothness to survive.
4. Label the 25 failure onsets by hand with `recording/label_onset.py`. Record your median disagreement with Gemini's. This is the exercise that unblocks the most.
