# Chapter 15. How decisions were made, and how they should be

This chapter is about judgement rather than results: not what was decided, but *how*, so that the next decision is yours. It covers the real decision history of the repository including the ones made by tools, the pattern of errors, and the division of labour that holds from now on.

## 15.1 The decision pattern that worked

Trace any good decision in the register and it has the same shape:

1. A gate was pre-registered with a bar and a decision rule.
2. The cheap baseline was computed first and often won.
3. The result was read for mechanism, not just verdict.
4. The decision followed the rule, including when the rule said "drop the thing you spent two weeks on".
5. The architecture spec was rewritten to match, and the outgoing version snapshotted.

Examples: H0 → reranking dropped the same day → v0.1 abandoned. G1b + diagnosis → planning dropped → regret adopted → G7 designed around object state. G6a/b → progress-by-distance dropped. The 10-19 fresh-rollout day → six mid-episode verifiers failed → v1.1 verifies at rest only (seven latent monitors by 10-27, with SURVEYOR). G-flora-abl → the commercial-rewarder plan replaced the non-commercial one.

The pattern's strength is that it is boring. Nothing in it needs insight; it needs the bar written down and the rule followed.

## 15.2 The decision pattern that did not

The errors of the three weeks, by type:

**Speed over understanding.** NEXT_ANGLES (fifteen entries in one evening). The weekly-scan → work-package → repo-writer chain built in one morning after the experimenter had said they wanted to slow down. The fifteen were nominally compliant (each had a decision line) and several were not decisions. The automation layers would have produced protocols faster than anyone could read them. The experimenter caught both, not the tools.

**Description from memory.** D-JEPA described as a loss term; it is an operator on executed outcomes. The description came from the assistant, from memory, and was executed. The method's step 2 (read the implementation section) was skipped by the person writing the plan, and the executor could not know. Caught by the executor when the protocol had to cite the paper.

**Stale facts carried forward.** "G9b running since 10-16" copied from a stale file into NEXT_ANGLES and EXPERIMENTS §3; the job had died a week earlier. Caught by the weekly pending-audit rule being applied for the first time.

**Conclusion before parity.** TurboVLA's two-week diagnosis concluded "ignores the instruction" before the port was compared to upstream. The data finding survived; the architecture claim did not. Caught when the experimenter suspected their own code.

**Verdict word too strong.** REAP written as *fail* with two live confounds. Caught in review; became *not proven*; became *fail* again the same day the confounds were removed by v2. The word was wrong the first time for the right reason: it was not yet earned.

**Trusting the instrument.** G11-A, G12. Caught the next day by a different script.

Every one of these has the same remedy, which is the method applied earlier in the pipeline: at planning time (read the paper; write one decision per evening), at protocol time (cite the section; check parity), at verdict time (the confound checklist), at bookkeeping time (the audit). None of them was a failure of effort; all were failures of order.

## 15.3 What the tools are for, and what they are not

**Claude Code on the Orin** builds instruments and runs procedures. It is good at: environment and dependency work (the two-environment path A build, fail-fast imports, worker caps), launchers with safety (deadlines, destroy timers, upload retries), parity scripts with shared noise, bootstraps and fold logic, register bookkeeping, finding its own arithmetic errors when a rule tells it to check (17 vs 13 TurboVLA failures; the teleop-as-success bug). It is bad at: knowing whether a plan's premise is true (D-JEPA), noticing that a verdict word is too strong unless asked, reading a paper for what it does not say, and stopping when the queue is long.

**The assistant in chat** is good at: structuring, cross-referencing the register against the files, drafting protocols from a correct description, writing. It is bad at: the same things, plus a bias toward producing more (three automation layers in a morning) when the honest answer was "nothing new today".

**The experimenter** is the only component that: chooses the question, reads the paper's method section, writes the bar with the decision in mind, reads the failure for mechanism, suspects their own code, and signs the verdict. These are not the steps that take the most hours. They are the steps that make the hours mean something, and they are the ones an interview tests.

The division that holds from now on, written as rules:

1. No protocol runs whose paper the experimenter has not read at D3 (reading list, Appendix B of METHOD.md).
2. No bar is written by a tool without the experimenter rewriting it in their own words first, or amending it with a reason.
3. No verdict reaches the site without the confound checklist (chapter 5) applied by the experimenter.
4. Tools build; tools run pre-registered procedures; tools do bookkeeping; tools stop at money and at the arm.
5. The weekly pending-audit is the experimenter's, by hand, with the files open.

**The split applied to the cross-layer faults (16.1c).** DECISIONS and LESSONS record what each fault changed, not who noticed it first, so the honest split is by the kind of probe, which is what the rules above govern. *Checks written into a procedure* and run by the tool: the strict-load check that exposed G11-A's random encoder, the hand count of fails against the parser that found the unescaped pipes (WS-039). *Controls written into a protocol* by the experimenter before the run: REAP v2's no-time / common-horizon / shuffled-action controls, TurboVLA's path A bar, C1's policy-identity baseline. *A rule applied after the fact*: G12's parts-must-sum, which voided a gate that had already driven a decision (D39). The first kind is rule 4; the second is rule 2; the third is why rule 5 exists.

## 15.4 Decisions still open, and how to make them

**The robot day.** Decision sentence per block (chapter 1 exercise 3). Power per comparison (chapter 7 exercise 3). The G7a block needs failures from a non-TurboVLA policy and ≥ 25 failure chunks before its bar can decide anything; if the cube rollout session cannot produce that, G7a on the day is a measurement, not a gate, and the runbook should say so.

**W2′.** Done 2026-10-23: trained ($0.65, seed 3073), probes identical to W2, on the Hub; lesson 24 closed.

**REAP.** Done 2026-10-23: v2 with the ramp target removed both confounds without the hand labels; fail at 0.79; action-conditioned. Next decision: retrain W1 with policy rollouts in the mix, or stop monitor work until the robot day produces them.

**Path A.** Done 2026-10-23: 0.020° vs control 0.105; six re-initialised tensors are the state projection and the action projection's last layer (width-dependent); fact 7 is the full story; correction row written; both posts rewritten.

**The prep queue.** Eleven bars written by the assistant. Rule 2 above: rewrite each in your own words before its run, or amend. Expect to reject two or three.

**Hardware.** No purchase until a frozen gate fails because of the hardware and the gate is named. The rigid table and camera mount are the exception: they remove a confound from every trial and cost little.

**Shisso.** The lab name is yours; the company name waits for a product. The site stays under your name. (Updated 2026-10-30: shissorobotics.com is now the main site; the thesis and structure are in 15.4a.)

**Applications.** The World Models RE posting fits your evidence better than the Data one (JEPA valued; "what you've built and shipped"; latent planning and MPC on real hardware is a question G1b and G10 answer with numbers). Apply to both; lead with the gates.

## 15.4a Decisions of 2026-10-27 to 10-30

**The thesis changed, because the register outgrew it.** The lab's original question was what a JEPA world model buys on cheap hardware. By 2026-10-30 the register held 21 papers across world models, VLA policies, failure monitors, reward models and judges, action chunking and real-time execution, memory and edge latency, and the site's thesis page was still the old blog index. The owner rewrote it (WS-060): *What survives when a paper leaves the lab?* Shisso re-runs published robot-learning claims under three conditions the papers rarely meet — **the hardware people can buy**, **the cheapest baseline first**, **the bar before the run** — with four verdict words and every correction dated. This is the same method as chapters 1–9, stated as a public promise; the book's title already said it. What changed is that the world model is now one family among seven, not the question.

How it was decided is the part to copy. The trigger was a measurable mismatch (three paper counts on three pages, eleven navigation items, the thesis describing a narrower project), not a feeling. The fix removed rather than added: four sections (Papers, Findings, Rig, Thesis), one register, one count generated at build, and the old thesis kept in a file for the record.

**Claims are written by hand, contradictions are computed (WS-061).** A model-based extraction pipeline, evidence levels and a new site stack were all proposed and refused. The reasoning: the value of a claim row is that someone who read the paper chose its words, and the value of a contradiction is that no one chose it. Chapter 2.6a has the result.

**No scale in LEGO kitting (D64).** The owner removed the scale from the kitting use case: the verifier becomes a camera-only tier-2 card (mask count and footprint at rest, wrist re-observe) checked against staged negatives. Reason: less hardware on the table, faster to a self-running cycle, and v1.5 had already made the verifier a plugin slot, so dropping one plugin changed no harness code. The pattern: an earlier architectural decision (D59, the verifier as a slot) made a later product decision cheap.

**Inference-time papers are timed first.** After AAC and PACT-VLA (chapter 11b), every protocol for a method that changes the inference cycle starts with the cycle on the Orin. Both were decided by their latency gate; the quality gates were still worth running, because PACT's offline pass is a result the register would otherwise not have.

## 15.5 How a research engineer decides under uncertainty

The pattern, distilled, for any future question:

1. What would I do differently depending on the answer? If nothing, don't ask.
2. What is the cheapest thing that could give the same answer? Compute it.
3. Write the bar before the number. Commit it.
4. Change one thing. Name what you could not control.
5. Check the instrument.
6. Count by the right unit; pair the comparison; state what n cannot resolve.
7. Read the failure for where the signal died, then suspect your own code.
8. Write it down once; correct by appending.
9. Decide, by the rule you wrote. Then rewrite the spec.

Nine lines. Everything else in this book is the reasoning behind each.

## 15.6 Exercises

1. For each error in 15.2, write the one-sentence rule that would have prevented it and the step of the method it belongs to.
2. Take the eleven prep-queue bars. Rewrite each in your own words without looking at the original. Diff. Where you differ, decide and commit the amendment with the reason.
3. Write the robot-day decision sentences and the power budget. If G7a does not survive, move it and say why in the runbook.
