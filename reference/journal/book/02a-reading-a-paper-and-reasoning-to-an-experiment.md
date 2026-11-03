# Chapter 2a. How to read a paper, and how to reason from it to an experiment

Chapters 1 and 2 give the method's first two steps as rules: start from a decision, translate the claim. This chapter is the reasoning behind them as it actually runs in an experienced research engineer's head, in the order it runs, with the questions asked at each point and what the answers do. It was written after reading 56 papers against this repository's gates, and every move in it was needed at least once to get one of those readings right — or was skipped once and cost a week (chapter 15.2).

The short version: **a paper is a claim, an instrument, and a number. Read the instrument before the number, the claim before the instrument, and decide what the number would change for you before any of it.**

## 2a.1 Before opening the PDF: the decision question

An experienced engineer does not read a paper to know what it says. They read it to answer one of three questions, and they know which before the first page:

1. *Does this change what I should build or run next?* (a method paper that competes with your current approach)
2. *Does this explain a result I already have?* (a mechanism paper; Planning Limits explained G1b a week after G1b failed)
3. *Does this set the standard I will be judged against?* (a protocol paper; coverage shift for judges, the blind-assembly figure for real-robot evaluation)

If none applies, the paper goes on the reading list at D1 (abstract and tables) and you move on. The weekly scan exists to sort papers into these three bins; the mistake it is designed to prevent is the one NEXT_ANGLES made on 10-22: fifteen "interesting" papers turned into fifteen protocols because the question "which decision does this change" was never asked.

Write the question down in one line before reading. It decides which sections you read and in what order, and it is the first line of the card you will write afterwards.

## 2a.2 The reading order, and why it is not front to back

**Abstract, then the main results table, then the experimental-setup section, then the method, then the related work, then the limitations, then the introduction last.** Each step asks something specific.

**Abstract (2 minutes).** Extract the claim as a single sentence with a subject, a comparison and a number: "a bounded set operator trained on executed outcomes raises Push-T from 83.6 to 87.9 and real PiPER from 56 to 74". If you cannot write that sentence, the abstract is hiding either the comparison or the number, and that is the first thing to note.

**The main table (5 minutes).** Before reading any text about it, answer four things from the table alone:
- What is the *unit* of the number? Success per episode, per task average, per trial, a proxy (AUROC, L1 error, Fréchet fidelity)?
- What is *n* per cell, and are there intervals? If neither is in the table or caption, find them in the setup section before trusting any difference between two cells.
- Which rows are *run by the authors* and which are *quoted* from other papers? SALT's Table 1 marks quoted rows with a dagger and says they are not comparable; many papers do not mark them at all. A quoted row is a different protocol.
- Which row is the *cheapest thing that could have produced the number*? Random, length, copy-last, the frozen base policy, the previous version of the same method. If it is absent, the paper has not shown the method does anything; it has shown the pipeline works.

**Setup / protocol (15 minutes, the section that matters most).** This is where the paper tells you what it actually measured. Read for: how many trials and seeds; what is held fixed between arms (checkpoint, manifest, planner, layouts); how success is defined and who labels it; whether evaluation states were seen in training; what was tuned on what (DeepJEPA says openly its threshold was picked on the evaluation sweep — most do not say); the hardware and what it can sense that yours cannot (torque, a second camera, a simulator reset). Write these down as a list, because the translation table (chapter 2) is built from it.

**Method (20 minutes).** Now the equations. The rule from chapter 15.2: read the implementation section, not the figure. D-JEPA's Figure 1 looks like a loss term; its §4 is a set operator with a bounded correction trained on executed outcomes. Those are different experiments. For each equation ask: what is the input, what is the target, what is detached, what is frozen, what is the one knob. Write the objective out in your own notation; if you cannot, you do not yet know what the model is trained to do.

**Related work (5 minutes).** Not for the citations: for the *baselines the authors chose not to run*. Related work names the competitors; the results table shows which ones made it. The difference is the list of things the paper does not establish.

**Limitations (5 minutes).** Authors are usually honest here and the sentence is usually the one you need: "we advance no mechanism, having measured none" (2610.19375); "transfer to pixel-scale models is untested" (DRPE); "a neural policy could lack this coupling" (coverage shift). The limitation you care about is the one that sits between their platform and yours.

**Introduction (last, 5 minutes).** By now you know what the paper did. The introduction tells you what the authors *think* it means, and the gap between the two is the paper's rhetoric. Reading it last stops the rhetoric from setting your expectations for the table.

Total: about an hour for a D3 read. A D4 read (one you will reproduce) adds the appendix and the code, and a parity target: one number you will match before you change anything.

## 2a.3 The tells: what experience notices in the first ten minutes

These are not rules; they are the pattern-matches that fire while reading, each learned from being fooled once.

- **n = 10 per cell on a real robot.** The Wilson interval on 7/10 is 40–89 %. Any difference under ~40 points between two such cells is unresolved. Say it; do not dismiss the paper, dismiss the comparison. (V-JEPA 2 Table 2, FOREWARN Table II, SEAR Table 4.)
- **No seeds on the real-robot table, seeds on the simulation table.** Honest about what they could afford; the real numbers are demonstrations.
- **A quoted baseline.** Different protocol, different data, different n. The matched reproduction (SALT's "reproduced LeWM", JEPA-WM's re-run DINO-WM) is the only row you can subtract from.
- **An oracle or upper bound in the table.** The strongest design pattern in the field: the real simulator as the perfect predictor (Planning Limits); true future latents (FOREWARN); a privileged verifier (VLA-Corrector); ground-truth quality labels (CUPID). A paper with an oracle can say *where the limit is*. A paper without one can only say its method beat its baselines.
- **An ablation ladder.** One change per row, cumulative, one metric (V-JEPA 2.1 Table 1). It is the cheapest evidence that each piece does something and it is rarely done on hardware.
- **"w/o X" under every table.** (VLA-JEPA's "without human video".) A one-row control that turns a model paper into an evidence paper.
- **The threshold was chosen on the test sweep.** If the paper says so, credit them and discount the headline by what a held-out threshold would cost. If they do not say where it was chosen, assume the sweep.
- **The matched control is missing.** Any auxiliary loss, any fine-tuning intervention, any "we added X": where is the arm that trains the same number of extra steps without X? 2610.19375 shows the control absorbing a 90 % "gain". If the arm is absent, the attribution is unsupported, however large and tight the number.
- **A proxy with no bridge.** L1 rollout error, Fréchet fidelity, probe r, one-step loss. Each paper that uses one should also show, somewhere, that it tracks the real outcome. RoboJEPA shows Fig. 5 beside Fig. 6; 2610.19375 shows r = −0.01 for the bridge everyone assumed. Ask for the bridge figure; if it is not there, the proxy is a hypothesis.
- **The failure is attributed to the data, not tested.** "Three-level fails on Push-T because episodes are too short" (H-JEPA) — plausible, untested. Mark as a hypothesis.
- **A failed preregistered test, reported.** (Coverage shift T1/T3; DeepJEPA's +1.1 pp with p = 0.42.) Raises credibility of everything else in the paper. Lowers it when a paper has no failures at all and five positive results.
- **"Over 50,000 evaluation episodes."** Total across platforms; per-cell n is in an appendix. Find it.
- **The thing your hardware cannot do.** Torque sensing (Imagine-RL, blind assembly), a simulator reset for counterfactual execution (D-JEPA), a fleet (Robo-COP), a third camera (ContactGuard). Not a flaw in the paper; the first entry in your translation table.

## 2a.4 From the reading to a decision: the ladder

Having read, the engineer does not ask "should I reproduce this". They climb a ladder of five questions, and most papers fall off before the top.

**1. What is the claim, in my units?** Rewrite it with your task, your n, your sensors. "Event-anchored imagined-latent probe reaches AUC 0.98 vs 0.84 for the current latent, on a torque-controlled Piper with three cameras and ~250 labelled grasps per task" becomes "on a position-controlled SO-101 with two cameras and 65 labelled episodes, would the imagined latent beat the current latent by ≥ 0.05 AUROC". The number in your version is smaller than theirs and you should say by how much (chapter 2's scale discount).

**2. What would change if it were true here?** One sentence. For ContactGuard: "W1 gets a monitor role and the robot day gets a pre-contact abort block". For Planning Limits: "nothing changes; it explains G1b and tells me to stop ranking against far goals". If the sentence is "I would know something", stop: that is reading, not an experiment.

**3. What is the cheapest test that could make it false?** Not the full reproduction. The one measurement that, if it comes out the wrong way, ends the question. For ContactGuard that was W1-C1: anchor, roll out, probe, compare to the current latent — one afternoon, $0, and it came out the wrong way. For TurboVLA it should have been the 40-frame task-swap on the released checkpoint *before* the two-week diagnosis. The cheapest falsifying test is usually a baseline or a parity check, which is why chapter 3 exists.

**4. What would I not be able to tell apart?** The confound list, written before the run (chapter 5): policy identity, episode length, elapsed time, the camera that moved, labels from a model, the threshold chosen on the test set. For each, either a control arm or a sentence in the protocol saying it is uncontrolled and the verdict can therefore be at most *not proven*.

**5. Is the answer worth the cost at the n I can afford?** Power (chapter 7): with 20 paired trials per arm you resolve ~30-point differences; with 69 episodes and 8 failures your AUROC standard error is ~0.09. If the claim's expected effect in your units is smaller than what your n resolves, the experiment is a measurement, not a gate, and the protocol says so — or you change the design to pool (21 tasks → n = 126 in the prep queue) or to pair.

A paper that passes all five becomes a NEXT_ANGLES entry. One that falls at step 2 becomes a card in the book. One that falls at step 5 becomes a "measurement, no verdict" line, which is still worth running if it is free.

## 2a.5 Deciding *how* to evaluate: the reasoning behind a protocol

Once the decision is real, the protocol is designed backwards from the verdict.

**Start from the sentence you want to be able to write.** "On 20 paired operator-staged layouts, top+side beat top+wrist by k/20 with a paired-bootstrap CI excluding zero, oscillation rate halved." Every noun in that sentence is a design choice: paired (so the noise of layouts cancels), operator-staged (SEAR: the harness's own resets avoid hard states), 20 (what the day affords), CI excluding zero (not a p-value, an interval), oscillation (REKI's criterion, so the number is comparable). Write the sentence first; the protocol is the list of things that make it writable.

**Choose the unit of analysis before the metric.** Episodes, not frames (frames are correlated); layouts, not trials (if layouts repeat); tasks, not episodes (if pooling). FARM truncates every episode to its task's common horizon so that length cannot be the unit's shortcut. Most disagreements about "is this significant" are disagreements about the unit.

**Put the oracle in if you can build one.** On your rig the oracle for any imagined-future test is the *real* future frame — you always have it offline. For any judge, the oracle is a human label. For any ranking test, the commanded joints. The oracle bounds what the learned thing could ever add; without it, "it works" has no ceiling to be measured against.

**Match everything but one thing.** Same checkpoint, same manifest, same seeds, same planner, same layouts; then vary the one factor (TD-JEPA's locked evaluation; JEPA-WM's one-component-at-a-time). If two things vary, name them both and accept *not proven* as the best possible outcome.

**Add the shortcut-removing controls that cost nothing.** Shuffled actions (does the head read the action?); zero actions; episode-length truncation; a random-init model with the same code (Planning Limits' 0.8 %, FARM's "random detector 0.650"); random-at-the-same-budget for any selection. Each is one line in a script and each has, in this repository, turned a *pass* into a *fail* at least once.

**Decide the verdict rule and the bar before the number** (chapter 4), and decide what *not proven* would look like — which confound, which n — so that the write-up after the run is a lookup, not an argument.

**Budget the instrument checks** (chapter 6): strict load, parity on one number, the noise floor on the same checkpoint twice, parts-must-sum. They are the first lines of the protocol, not an afterthought, because they are the ones that void gates (G11-A, G12).

**Write the cost and the kill switch.** Pilot, measure seconds per step, multiply, add the deadline. Say what the run costs if it is wrong.

## 2a.6 Three readings, end to end

**ContactGuard (2608.13438), read 10-23.** Decision question: does W1 get a monitor role? Claim in my units: imagined-latent probe ≥ current-latent + 0.05 AUROC on 65 episodes, two cameras, position control. Cheapest falsifier: W1-C1, one afternoon. Confounds listed: teleop-only world model vs theirs on policy rollouts; 65 vs 250 labels; two vs three views; torque vs position. Result: 0.70 vs 0.82, opposite sign. Verdict: *fail* for the transfer; mechanism unknown; the paper's own Table 3 (shuffled chunks) named as the next $0 control. What changed: W1 stays a logged signal; the monitor question moved to "what W1 is trained on". One afternoon, one line in the register, one blog post.

**Planning Limits (2609.39235), read 10-23.** Decision question: does this explain G1b? Claim: the plannable range equals the rollout length; a *real simulator* falls 92 → 41 % when the goal is 20 steps past a 5-step rollout. In my units: G1b scored 7-step chunks against goals many chunks away. Step 2: nothing to run; the mechanism is established with an oracle I cannot build. What changed: the G6/G7 goal definition (a demo frame ≤ 1 chunk ahead), one new angle (§19, VLA+WM selection, which the paper shows with no-task-worse evidence), and a sentence in the G1b post. One hour of reading replaced a week of re-running G1b with a "better" model.

**CUPID (2506.19121), read 10-23.** Decision question: which demonstrations help, and is it the same as which look good? Claim in my units: influence on closed-loop return, from ~25–50 labelled rollouts, ranks demos; beats quality metrics where the data holds a brittle strategy or a spurious correlation. Fact 7 *is* a spurious correlation (pose ↔ target). Cheapest falsifier: curation accuracy of the scores against my own labels, offline, before any retrain. Confounds: the retrained policy differs in more than data unless budget and seed are fixed; so four arms at matched budget, random and length as baselines. Power: 20 paired layouts per arm resolves ~30 points — so the bar says "≥ 2/20 over both baselines, CI excluding zero", and anything smaller is a measurement. Became §24.

## 2a.7 The habits that make this fast

Experienced is mostly *fast at the same checks*, not different checks. The ones that save the most time:

- Keep a one-page card per paper in a fixed shape (claim in your units; protocol facts; baselines not run; the one number to parity-check; the cheapest falsifier; what it would change). The "a" chapters are these cards at length; the weekly digest should produce them at a paragraph.
- Keep a running list of **controls you have been fooled by**: random-init encoder (G11-A), stand-in model (G12), time feature (REAP v1), Gemini onsets (REAP v1), the port (TurboVLA), length (G9b), seen rollouts (G10). Read every new protocol against the list before the run. The list is chapter 15.2, and it should grow.
- Translate scale explicitly (chapter 2's table): their n, their data hours, their GPU-days, their sensors; your numbers beside them; the discount applied to the expected effect.
- Say the n sentence out loud for every number you cite: "0.70 vs 0.82 on 65 episodes, so the difference is about one and a half standard errors" — if you cannot say it, you do not yet own the number.
- Prefer the free experiment that removes a confound over the paid experiment that adds a method. REAP v2 cost nothing and turned *not proven* into *fail*; path A cost $2 and withdrew a claim; neither added anything to the stack, and both are the results you will be asked about.
- Read the paper's failures as carefully as its successes; they are where the method's edge is and where your hardware usually sits.

## 2a.8 A reading template

```
Paper:            (title, arXiv, date read, depth D1–D4)
Decision question: which of the three (build/run | explain | standard), one line
Claim, theirs:    subject – comparison – number – unit – n – platform
Claim, mine:      same sentence in my units, with the expected discount and why
Protocol facts:   n per cell | seeds | intervals | held fixed | varied | labels by | tuned on | sensors
Baselines run:    ... | not run: ... | oracle/upper bound: yes/no
Proxy and bridge: proxy used | figure showing it tracks the outcome | or "none"
Tells:            (from 2a.3, any that fired)
Cheapest falsifier here:  one measurement, cost, time
Confounds I cannot remove: ...  → best possible verdict
Power at my n:    effect resolvable | so: gate | measurement
What it changes:  one sentence; or "nothing — card only"
Parity target:    the one number I will match before changing anything (D4 only)
```

## 2a.9 Exercises

1. Take a paper from this week's digest you have not read. Fill the template in one hour, in the reading order of 2a.2, without reading the introduction until last. Then read the introduction and write down what it claimed that the table did not support.
2. For ContactGuard, Planning Limits and CUPID, redo the five-question ladder (2a.4) from the PDFs without looking at 2a.6; compare; where you differ, decide who is right and why.
3. Re-read your own TurboVLA post of 10-08 as if it were a submitted paper. Apply the tells of 2a.3 to it. List the ones that fire. (At least three should.)
4. Add three entries to the "controls I have been fooled by" list from the last two weeks that are not already in 15.2.
