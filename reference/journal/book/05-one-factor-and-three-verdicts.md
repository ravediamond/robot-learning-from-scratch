# Chapter 5. One factor at a time, and the three verdicts

## 5.1 The principle

An ablation changes one thing. When two things change between the arms of a comparison, the comparison tells you about neither.

When a confound cannot be removed, the verdict is **not proven**, not **fail**. The three verdict words have exact meanings and are not interchangeable:

- **Pass**: the pre-registered bar was met.
- **Fail**: the bar was missed, and no identified confound could explain the miss.
- **Not proven**: the bar was missed, and a named confound could explain it. The method may work; this test cannot tell.

Using *fail* for a confounded result is the most common way a working method gets thrown away. Using *pass* for a confounded result is the most common way a broken one gets kept.

## 5.2 Why: what a comparison can carry

A two-arm comparison carries one bit of information per factor that differs. If arm A and arm B differ in the model *and* the input format, a difference in outcome is attributable to either or both, and the experiment has told you "something in this set of two changes matters", which is rarely what you wanted to know. The cost of an extra arm (a third run that changes only one of the two) is usually small next to the cost of a wrong conclusion; the Flora ablation cost ~$4 and settled a question the main run had left open.

Confounds are factors you did not vary on purpose but that differ between the groups being compared. In robot evaluation the recurring ones are: which policy produced the rollouts; who labelled the outcomes and how; when the episodes were recorded (lighting, operator fatigue); which layouts were used; and the sample size itself, which makes a difference of any size "not significant" and tempts the word *fail*.

## 5.3 Worked cases: clean ablations

**Flora vs its base (G-flora-abl, 2026-10-20).** The question "is it the robot fine-tune or the input format" was answerable only because the input was held fixed: the same 64-frame side-by-side clip and the same terminal prompt went to Flora-27B, Qwen3.5-27B (its base), Qwen3.5-9B and Gemini Flash. Results: 8/8 at 2/42; 7/8 at 2/42; 16/42 successes flagged (fail); 8/8 at 3/42 (narrow fail). One factor at a time (fine-tune on/off at fixed size; size at fixed input; model family at fixed input) gave three separate conclusions: the fine-tune adds little; size matters at this task; the format did the work. Had the ablation also changed the prompt for the base model, none of the three would be available.

**H0 (2026-10-08).** Reranking vs the mean, over the *same* eight samples. The mean is not just a baseline; it is the ablation that removes the selection step while keeping everything else.

**G13b (2026-10-19).** Prefix overwrite vs exact RTC vs naive, on the *same* 254 boundaries, the *same* checkpoint, the *same* 20 episodes. 72–74 % vs 21–41 % vs 0. Then the vla.cpp port of prefix overwrite vs the PyTorch version: 71.0 / 73.3 / 71.5 % vs 72.8 / 74.0 / 71.7 %, prefix error 0.0, default path bit-identical. The second comparison is the parity check (chapter 6) framed as an ablation: the only factor changed is the engine.

**The one-step-fidelity paper's lesson for §7 (2610.19375).** A grounding head is compared against a model trained for the same *additional* number of steps without the head. Otherwise "the head helps" is confounded with "more training helps". The matched continued-training control is now written into §7 before its run.

## 5.4 Worked cases: confounds that forced "not proven"

**C1 (2026-10-11).** Does the imagined latent predict rollout failure better than the current latent? 81 rollouts, 66 ok / 15 fail. Policy identity alone: 0.76. Current latent: 0.68. Imagined: 0.72. TurboVLA produced most of the 15 failures. Any probe that learns "this looks like a TurboVLA rollout" scores near 0.76 without knowing anything about failure. Verdict: not proven. The register says so. It would have been easy to write "fail: imagination does not help", and the world model's monitoring role might have been dropped two weeks early on a confounded result.

**REAP (2026-10-23).** No monitor meets "≥ 5/8 failures at ≤ 1/20 false interventions". The logistic risk head: 6/8 caught at 7/42 false interventions, AUROC 0.93. Two confounds: (1) on the 69-rollout secondary set it scores 23/25 at 4/44, but 17 of those 25 failures are TurboVLA, which fails early and visibly; (2) the onsets were labelled by Gemini, and on the 5-episode spot check two missed-grasp episodes had the onset placed at a later retry, 110 and 289 frames late. A risk head trained to fire "within 1 s of onset" against onsets that are sometimes 10 s late is being scored on a wrong clock, and its false interventions may be true warnings. Verdict at the time: not proven pending hand-relabelled onsets. (The first write-up said *fail*; this book is where the correction was argued.)

**REAP v2 (2026-10-23, evening) — the confounds removed.** The paper's own protocol: ramp target (0 on successes, t/(T−1) on failures, so no onset labels at all), no time feature, every episode cut to the task's common horizon (374 frames) so elapsed time cannot be a shortcut, shuffled-chunk and zero-action controls, bTWA, a parameter-free GRU as the floor. Protocol committed before the run. Primary 50, medians over five seeds: logistic head **3/8 at 5/42, AUROC 0.789**; MLP 6/8 at 13/42; GRU floor 2/8 at 3/42. The 0.93 was the time feature and the episode length. Verdict: **fail**, clean. The one thing that survived is the action-conditioning test (true vs shuffled chunk +0.08, above the +0.05 line): the head reads the commanded action; it does not read enough about failure. This is the pattern to remember: a *not proven* with a named confound became a *fail* the same day the confound was removed, at $0, and the 25 hand labels were never needed. Naming the confound was the whole value of the verdict word.

**TurboVLA (2026-10-08 to 10-23).** "TurboVLA ignores its instruction on two-task bowls" was the conclusion of two weeks of careful probing. Then the port was compared to upstream: the fusion module was random-init (the paper initialises it from Grounding DINO), structurally different (no text self-attention layer, no layer-scale), and had the state token inside fusion (the paper feeds state to the decoder only). The released checkpoint with upstream code passes a task-swap control (0.105 vs 0.095) that the port fails 20–55× over. Fact 7 (the data property) stands; the architecture conclusion was *not proven* until path A reported. **Path A (2026-10-23):** upstream trainer, full LIBERO init (666/672 tensors), 35k steps, $2.0 — task swap **0.020°** against the 0.105 control. The faithful model ignores the instruction on this data too. So the port was a real confound that, once removed, changed nothing: fact 7 is the full story, and the two-week diagnosis was right about the data and wrong to say anything about the model before parity. This is the confound you least want to find — your own implementation — and the reason the order is parity first, diagnosis second.

**HWM (2026-10-28): two failed bars, verdict not proven.** H1 (a crossover between 1 and 2 s) and H2 failed as pre-registered; H3 met. The written verdict is *not proven*, by the owner, because the run has a confound it cannot separate: ~0.6 h of one task against the paper's ~130 h (12b.8b). This is the hardest case in the register to defend and the one most likely to be asked about. What makes it defensible: the bars were not moved and the numbers are printed as the bars say; the confound is chapter 2's scale rule, nameable before the run; the experiment that would separate the readings is named. What makes it weaker than C1: the conditional verdict ("not proven if the data is ≥ 100× smaller") should have been in the protocol. Rule added in chapter 4.5a: **write the conditions under which a failed bar reads not proven into the protocol, before the run** (a verdict map).

**Horizon × depth (2026-10-29): between the bar and the fail line.** A1 required R(2)/R(10) ≤ 0.70 and named 0.85 as the fail line; the result was 0.78, CI 0.73–0.81, entirely between the two. A protocol with a pass line and a separate fail line has a third region by construction, and its name is *not proven*. The arm gate was conditional on A1 and was not run (11b.3).

## 5.5 The confound checklist

Before writing a verdict, answer each:

1. **Policy mix.** Were the rollouts produced by more than one policy? If so, what is the result on the single-policy subset, and does one policy dominate the failures?
2. **Label source.** Who labelled outcome/onset, and what is their measured error (the 5-episode spot check, the +6 median offset, the two 110/289-frame misses)?
3. **Sample.** Is n large enough that the miss could not be noise? (Chapter 7.) A difference whose CI includes zero is not a fail; it is "cannot tell at this n".
4. **Split leakage.** Did any frame of a test episode influence training, including through feature extraction, normalisation statistics, or a probe fit on all episodes? (G10 was retrospective on seen rollouts; its thresholds did not transfer to fresh ones. That is leakage of a mild kind: the model had seen the videos as unlabelled data.)
5. **Implementation.** Is the thing you tested the thing the paper describes? (TurboVLA. D-JEPA's loss-vs-operator.) Parity against reference code where it exists.
6. **Measurement.** Did the instrument measure the model? (Chapter 6: strict load, parts-must-sum.)
7. **Task property.** Does the task require what the method provides? (Chapter 2: instruction redundancy, ceiling.)

If any answer is "yes, and I cannot remove it", the verdict is *not proven* and the write-up names the confound and what would remove it.

## 5.6 Why "not proven" is not a cop-out

It names a specific next experiment. "Not proven because policy identity dominates" means: collect failures from the policy you care about. "Not proven because onsets are VLM-labelled" means: hand-label 25 onsets. "Not proven because the port diverges" means: run the faithful version. Each is a concrete, costed step. "Fail" names nothing and closes the line.

The symmetric error is as bad: calling a confounded *pass* a pass. G10's AUROC 1.000 was on seen rollouts with thresholds fit on the same data; the honest word for it at the time would have been "not proven until fresh rollouts", and the fresh-rollout day (W1-A) is where it should have been expected to shrink. The architecture was rewritten around G10 for two days before that check. Not expensive, but the kind of thing that, with a bigger team and a longer cycle, becomes a quarter of work on a result that was never there.

## 5.6a The other words: withdrawn and void

Three verdicts describe a result. Two more describe the record, and the site uses all of them:

- **Withdrawn**: a verdict was published and then found to rest on a broken instrument or a wrong implementation. TurboVLA is the case: "ignores its instruction" was a statement about the port; the paper verdict is withdrawn, the data finding stands separately, and the correction row is dated. Withdrawn is not a fail of the paper.
- **Void**: the protocol's own measurement check failed before any result was read, so there is no result. PRIME (13a.10a, held-out monotonicity 0.51 < 0.8) is the case; its rollout scores had been computed by the script and are reported nowhere. G11-A and G12 (chapter 6) are voids found after the fact.

The site's verdict vocabulary is four words per paper — reproduces, fail, not proven, withdrawn; *void* applies to a run, not a paper. At 2026-10-30 the 21 papers in the register read: 4 reproduce, 12 fail, 3 not proven, 1 withdrawn, 1 not yet tested.

## 5.7 What the interviewer will ask

*"Your monitor failed. Why do you think it might still work?"* "I don't, any more. v1 was not proven with two named confounds; v2 removed both with the paper's own protocol and it failed cleanly at 0.79. What survived is that the head is action-conditioned, so the next version changes the world model's training data, not the head."

*"Give me a result you reported as a fail and later found was confounded."* TurboVLA. Be direct: "I concluded the architecture ignored instructions; the port I tested was not the architecture. The data finding survived; the architecture claim did not. I corrected the register and the post."

*"How do you decide between 'fail' and 'not proven'?"* The checklist. "If I can name the confound and the experiment that removes it, it's not proven. If I can't, it's a fail."

## 5.8 Exercises

1. Run the 5.5 checklist on every verdict in `EXPERIMENTS.md` §2. List the rows whose verdict word should change. (There are at least two.) Write the correction rows.
2. Design the third arm that would have been needed to make G9a-AB a clean comparison between Robometer and episode length (hint: a length-matched subset).
3. For REAP, write the v1 → v2 story as it would appear in a blog post: the v1 table, the two confounds, the v2 protocol that removed them, the v2 table, and the one sentence on what the verdict change teaches.
