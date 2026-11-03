# Holding the world-model view under pushback (16.7b), as dialogues

Each answer is under 60 words. Say it aloud; if an answer runs long, cut the clause after the number. The order inside dialogue 1 is the whole trick: agree from your register first, then the narrower claim.

## Dialogue 1 — the data pushback (PI-style, 16.1c)

**Push 1.** "The model class is a downstream choice. What matters is the data. Why spend time on world models?"
**Answer 1.** I agree, and my register says so. Every result that mattered was decided by what the data contained: an action-only baseline out-ranked the world model at planning, G1b, 0.60 against 0.68–0.90; and TurboVLA ignored its instruction because the instruction was redundant on 94 % of the frames.

**Push 2.** "Then you've conceded. Why keep the world model at all?"
**Answer 2.** Because the question I can test on a low-cost rig isn't which model, it's which role a world model earns given the data I can afford. The answer was role-specific: verifier at rest, yes — G10, AUROC 1.0. Planner, no — G1b. Monitor, no — W1-A and REAP v2 failed.

**Push 3.** "A verifier at rest is just a classifier on the last frame. Your colour masks do that."
**Answer 3.** Right — and the deterministic judge wins at rest: G9c-3, zero false positives in 25. The world model's verifier role stays in shadow mode; it logs beside the judge and never decides. That's exactly the data-conditioned answer: on this task the cheap instrument is enough.

**Push 4.** "So what is left for the world model?"
**Answer 4.** Prediction before the outcome is visible. Offline it predicts failure 30 steps ahead at +0.179 over policy identity, G10's shadow veto. It isn't deployed: it needs a fresh-rollout gate against joints only, because G1b showed the joints explain a lot on a position-controlled arm.

**Push 5.** "Your monitor results failed. Isn't that the data problem again?"
**Answer 5.** Yes. C1's probes lost to policy identity alone, 0.76 — the failures belonged to one policy. REAP v1's 0.93 fell to 0.79 once time and episode length were removed. Both are data findings dressed as model gates, and I say so in the register.

**Push 6.** "Then why would you ever need a world model?"
**Answer 6.** Your own position needs one below the semantic level — contact, object state. That's where my next gate sits, the object-state test, where decoys share the arm state and the joints can't separate them. If the world model fails there too, I drop it.

**Push 7.** "Why JEPA rather than a video model for that?"
**Answer 7.** Cost, measured where it can be. The one matched study, LaLa-World, same encoder and data: the latent model planned better, 59.4 against 48.0, after six times less training and 13.9 times faster — in simulation. On a Jetson that's the difference between testable and not.

**Push 8 (your turn).** —
**Answer 8, the question back.** What is the world model's role on your robot today — planner, monitor, verifier, data filter — and which experiment decided it?

## Dialogue 2 — the LeCun-style push ("JEPA is the only path to real understanding")

**Push 1.** "Generative models can't learn real world models. JEPA is the only way. Do you agree?"
**Answer 1.** I don't need that claim for my results. On this rig a JEPA model learned the arm's dynamics, G0, 0.31 against copy-last's 0.89, and still lost at planning to the commanded joints. Whether it is the only path isn't something I tested.

**Push 2.** "Then your planning failure is a scale problem. Train bigger."
**Answer 2.** Possibly, but RoboJEPA's encoder already takes 526 ms for two views on the Orin against a 200 ms bar. The scaling path leaves the robot's own computer. That's a hardware fact, not an argument against the architecture.

**Push 3.** "You're blaming the hardware."
**Answer 3.** I'm stating where the claim was tested. The planning loss is about position control: decoys differ in arm state, so joints separate them. A bigger model doesn't change what the joints already know. The test that would is object-only decoys.

**Push 4.** "What would convince you JEPA plans?"
**Answer 4.** Top-1 at or above 0.8 on chunks that share the arm state and differ only in the object's outcome, on held-out episodes, against the action-only baseline. Pre-registered, one factor. If it passes there, I'd move it from verifier to planner.

**Push 5.** "Your latents aren't monotonic in progress. Doesn't that kill planning?"
**Answer 5.** It killed the progress signal: H-JEPA's slow features gave ρ −0.73 against a −0.80 bar and 52 % false stalls, G6b. The arm's trajectory loops, so distance to the goal frame rises mid-episode. A cost that ignores the loop is the fix to test, not more abstraction.

**Push 6.** "Then hierarchy is the answer?"
**Answer 6.** Not on my data: the second level made false stalls worse. Hierarchy needs factors at separated timescales; a 25-second bowls task barely has them. A longer task would be the fair test.

**Push 7.** "So JEPA has failed you?"
**Answer 7.** As a planner and a monitor here, yes. As a verifier at rest it reads outcomes perfectly. That's a role, not a verdict on the family.

**Push 8 (your turn).**
**Answer 8, the question back.** Which of those roles do you need a JEPA model for, and what would you accept as the test that it earned it?

## Dialogue 3 — "Why not just a video model?"

**Push 1.** "Video models already predict the world. Why build a latent predictor?"
**Answer 1.** Because I can afford to test it. Pixel generation on a Jetson isn't a live option; a 15M-parameter latent predictor is. LaLa-World is the one matched comparison: same encoder and data, the latent model planned better, 59.4 vs 48.0, 13.9 times faster.

**Push 2.** "That's simulation."
**Answer 2.** Yes, six steps or fewer, one study. I quote it as evidence for the efficiency argument, not as a verdict on the families. My own numbers are on the arm and say less about families than about roles.

**Push 3.** "A video model could also verify outcomes."
**Answer 3.** It could, and so does a colour mask: G9c-3, zero false positives in 25. The cheapest instrument that's right wins at rest. The interesting comparison is prediction before the outcome, where neither has been tested fresh on my arm.

**Push 4.** "Video models give you data augmentation for free."
**Answer 4.** Generated data has to pass the same bar as recorded data. My data findings — the 94 % redundant instruction — are about what the labels carry, and generated frames inherit the labels. I'd test it with the selected-versus-random protocol before training on any of it.

**Push 5.** "Wouldn't a video model catch your TurboVLA problem?"
**Answer 5.** No — that wasn't a perception problem. The authors' own trainer on my data ignored the instruction too, 0.020°. The fix is layouts where the pose doesn't decide the side, which is a data-collection decision.

**Push 6.** "So what would you use a video model for?"
**Answer 6.** Off-board, for evaluation: imagining a policy's rollout to pre-screen it before the arm runs it. That's an evaluation tool, and it would need its own gate against recorded outcomes before I trusted it.

**Push 7.** "And on device?"
**Answer 7.** Nothing yet. The Orin runs SmolVLA at 238 ms per chunk; there's no headroom for pixel generation in the control loop.

**Push 8 (your turn).**
**Answer 8, the question back.** Where does the video model sit in your loop — training data, evaluation, or control — and what did it replace?
