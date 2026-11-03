# Chapter 8. Reading a failure

## 8.1 The principle

A fail tells you the bar was missed. It does not tell you why. The number is the start of the work, not the end: a fail with a mechanism is a result; a fail without one is a cost. The craft is to locate, with probes, where the signal you expected dies, and to state the mechanism in a form that generalises beyond the one system you tested.

## 8.2 The probe chain

A learned system is a pipeline: input → representation → fusion → readout → action → outcome. When the outcome is wrong, the signal was lost somewhere along it. A probe is a small, cheap measurement at one stage that asks "is the information here?" The chain is: walk the pipeline, probing each stage, until the first stage where the answer changes from yes to no. That stage is the mechanism.

Three kinds of probe, from cheap to expensive:

- **Intervention probes.** Change one input, measure the change in output. Task-swap (change the instruction, measure the action delta); action-shuffle (shuffle actions, measure the world-model error); image-zero (black the cameras, measure the action). Minutes each, no training.
- **Linear probes.** Fit a linear model from a stage's activations to a quantity you care about (the task id, the arm pose, the outcome). Tells you whether the information is *present* at that stage, regardless of whether it is *used*. Minutes to fit.
- **Gradient and attention probes.** Where does the loss's gradient flow; where does attention put its mass. Tells you what training *cares about*. Minutes, one backward pass.

## 8.3 The worked case: TurboVLA, six probes

The register's facts 1–8, read as a chain:

1. **Rollouts fail (0/10, 1/10).** First suspicion is always the pipeline. Normalisation, scaling, colour order, task plumbing, camera order, calibration, latency: each checked, each ruled out. (The "ruled out" list in `projects/turbovla/README.md` §0 is longer than the findings list, which is correct; most of the work of a diagnosis is elimination.)
2. **Intervention probe: task-swap.** Change the instruction on the same frame. SmolVLA moves 7–13°; TurboVLA 0.1–0.25°. The instruction is ignored. Per-layout probe (does "left bowl" steer further left): 42–64 % (chance) vs 86 % for SmolVLA. Stage located coarsely: somewhere between text and action.
3. **Is the text path dead?** Attention probe: the decoder puts 7–21 % of cross-attention mass on text (3–10× uniform). Gradient probe: text tokens get 24–33× the per-token gradient of image patches. Linear check: the colour token differs 46–63 % between instructions after projection. The text path is alive and training cares about it. Stage narrowed: downstream of text encoding.
4. **Where in fusion?** Per-layer intervention: in fusion layers 2–5 the instruction-specific part of the text→vision update is 1–4 %; the final action chunk differs ~0.5 % between instructions. The signal is washed out in fusion.
5. **Could the decoder read it if it wanted to?** Auxiliary linear head on the decoder's final states, trained to predict "which bowl": 98 % in 1k steps. So the task *is* represented where the action head reads. Represented, not used. The stage is now exact: the action readout ignores a signal that is linearly available to it.
6. **Why would training not use it?** This is the step from "where" to "why". On paired pre-grasp training frames, swapping the instruction changes the L1 loss by 0–4 %. Why so little: the side decision (the 7° pan toward the target bowl) happens in a band of 0.1–0.3 normalised pre-grasp progress, ~20 % of pre-grasp frames, ~6 % of all frames; before the band the target is identical for both tasks (lift); after it the arm's own pose already points to the target. So the 0.4 s L1 target is predictable from the state almost everywhere, and training barely rewards reading the instruction. The cheapest fit is copycat: continue the arm's current direction. On the arm, the direction at the decision point is set by small motions, hence the coin flip and the hesitation.

Fact 7 is the mechanism, and it generalises: *any* policy with a proprioceptive state input, an action space close to the state space, and a short L1 target on teleop data where the instruction-dependent decision is brief, is at risk of this shortcut. That is a claim about data and objectives, not about TurboVLA, and it predicts things (SmolVLA at chunk 12 should also lose the instruction; a 50-step horizon should recover it) that can be tested.

## 8.4 Then suspect yourself

The chain above was complete and the conclusion ("TurboVLA ignores instructions on this data") was published. Two weeks later the port was compared to the upstream code: three divergences in the fusion module (random init instead of Grounding DINO weights; a different block without text self-attention or layer-scale; the state token inside fusion instead of at the decoder). The released checkpoint with upstream code passes a task-swap control (0.105 vs 0.095) that the port fails 20–55× over.

Fact 7 stands: it is about the data and the objective, and the faithful reproduction will measure whether a correctly initialised fusion stack overcomes it. The architecture claim does not stand until then. Step 7 of every probe chain is: *is the thing I tested the thing I think I tested?* Parity with the reference (chapter 6) is how you ask it. It belongs at the start of the chain, but when it was skipped, it belongs at the end before publication.

**What the faithful run decided (2026-10-23).** Path A — the authors' own trainer, full LIBERO init, on the bowls data, against a pre-registered bar — read the instruction no better than the port (task swap 0.020°, per-layout 38.9 %). The TurboVLA correction row withdrew the port claim and kept the data claim. The fault presented at the engine (our port) and lived in the data; it is a row of 16.1c's table, the same list seen as hiring evidence, and the reason 8.7 no longer calls it a port fault.

## 8.5 Other failures read in this repo

**G1b.** Top-1 0.60. Probe: pairwise accuracy 0.92–0.93; 0.92^7 ≈ 0.56. Mechanism: the metric compounds. Then: the action-only baseline at 0.68–0.90 with regret ≈ 0. Mechanism: the arm is position-controlled; the decoys differ in arm state; the joints know. Generalisation: latent planners on position-controlled arms must be tested on object-state differences.

**G6a.** Monotonicity −0.32 to −0.50. Probe: plot latent distance over a successful episode. Mechanism: the arm's trajectory loops (approach, lift, swing, place), so distance to the final frame rises and falls. Generalisation: raw latent distance is not progress on any task with non-monotone trajectories, which is most manipulation; a progress signal needs progress labels (TD-JEPA's 0.65 vs 0.95).

**G16-B.** VLM checkpoint judge flags 41/42 successes. Probe: look at the frames at the two checkpoints. Mechanism: the post-grasp checkpoint fires before the lift (the bowl is held but not raised); at rest, nested bowls look like "blue alone on the mat" from the top and a pink object near the base from the wrist. Generalisation: VLM judges need the whole episode and both views at the end, not snapshots mid-task; G-flora confirmed it.

**W1-A.** The live reflex fires on 42/42 successes. Probe: per-check distribution of surprise on successes: 23 % exceed 2.2 at k = 3, 20 % exceed 10 at k ≥ 15. Mechanism: the thresholds were fit on seen rollouts (G10) and the fresh distribution is wider; and the two-camera USB timing jitter shifts the scale. Generalisation: thresholds fit retrospectively do not transfer; calibrate on fresh data or use a conformal construction (and then W1-C2 showed the naive conformal band is not conformal across time when μ, σ are fit on the same successes).

**REAP.** The risk head at 7/42 false interventions. Probe: which successes does it flag, and when relative to the Gemini onset? Partial so far: two onsets were 110 and 289 frames late. Mechanism established by v2 (chapter 5.4): removing the time feature and cutting to the common horizon dropped AUROC 0.93 → 0.79, so the signal was elapsed time; the chunk-shuffle control shows the head still reads the action (+0.08). Verdict fail. Where the signal died: in the world model's knowledge of this policy's failures, which it never saw in training.

## 8.6 Writing the mechanism

The form that makes a fail publishable:

> *What failed:* [metric, number, bar].
> *Where the signal dies:* [stage, with the probe that showed it and its number].
> *Why:* [the mechanism, in one paragraph, in terms of data, objective or hardware, not in terms of "the model is bad"].
> *What it predicts:* [one or two testable consequences].
> *What would fix it, and at what cost:* [the next experiment].
> *What it does not say:* [the scale, task or implementation caveats from chapter 2 and 5].

The TurboVLA README §0 has all six for the data finding and now needs the last line updated for the port finding. The G1b diagnosis has all six. G9c has them in a compressed form. Most fails in the register have the first and the last; the middle four are what turn a register row into a post.

## 8.7 What the interviewer will ask

*"Tell me about a time a model didn't work and what you did."* TurboVLA, as a six-step chain, each step with its number, ending on fact 7 and its generalisation. Then the self-suspicion step. Two minutes.

*"How do you tell a data problem from a model problem?"* Intervention probes first (they are free): swap the instruction, shuffle the actions, zero the images. If the output does not move, then linear probes: is the information present at each stage? If present but unused, look at the objective: what does the loss reward? "Represented but not used" is almost always an objective/data mechanism.

*"What is the lowest level you have had to go to when a result was wrong?"* The connector question (16.1c). Not the TurboVLA chain — that one lived in the data, and the port was cleared by path A. The G11-A encoder (`strict=False` left it at random init: a loader fault), G12 (a benchmark fault: parts did not sum), the register row whose unescaped pipes shifted its cells (a file fault), the five source builds on the Orin (a platform fault). Say which probe found each, in one clause, and that in every case you fixed the layer and re-ran before touching the model.

*"What would you do with ten times the compute?"* Not "train bigger". Run the matched controls the mechanism predicts: SmolVLA at chunk 12 (should lose the instruction), TurboVLA at chunk 50 (should recover it), state dropout (should recover it). Those are the tests of the mechanism, and they are cheap at any scale.

## 8.8 Exercises

1. Reproduce the task-swap probe on the 21-task SmolVLA for three instruction pairs. Report the action delta in degrees with the per-frame distribution. Then do it with the images zeroed.
2. Write the 8.6 form for G6a, W1-A and G16-B from the register and the reports. Three pages.
3. Path A reported 0.020° (fail against the 5° line, control 0.105). Write the paragraph you *would* have written for each outcome before you knew, then the one that is true. Say which of your two predictions was closer and why.
