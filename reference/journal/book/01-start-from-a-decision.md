# Chapter 1. Start from a decision

## 1.1 The principle

Before reading a paper closely, before writing a line of code, write one sentence of the form:

> If this passes, I will **X**. If this fails, I will **Y**.

If X and Y are the same action, do not run the experiment. If you cannot write the sentence, you do not yet know why you are running it.

This sounds trivial. It is the rule most often broken, by experienced people, for the most respectable reasons ("it would be interesting", "the paper is hot", "we should know"). It is first because it is the cheapest filter: it costs a sentence and it removes the majority of candidate experiments before they cost anything else.

## 1.2 Why: the economics of a one-person lab

A one-person lab — one engineer with limited hours, one arm, two robot days a week — has roughly these budgets per month: 8 robot days of ~6 usable hours; perhaps 40 evening hours of attention; a few tens of dollars of rented GPU; and, hardest to replace, a finite ability to read results carefully. Every experiment draws on all four. The only product of an experiment is a change in the next action. An experiment whose outcome does not change the next action has consumed budget and produced nothing, however elegant its result.

This is a stronger statement than "prioritise". It says that an experiment with no decision attached has *negative* value, because it displaces one that has, and because its result will sit in the register looking like knowledge.

The industrial version of this rule is familiar to anyone who has run A/B tests: "what will we ship if it wins, and what if it loses?" Teams that cannot answer run tests whose results are argued about afterwards. The research version is identical; the stakes are your own time.

## 1.3 The two kinds of experiment

**Gates** have a bar and a decision: pass → X, fail → Y. G0 (is the world model real: pass → build on it, fail → fix training first). G1b (can it rank chunks: pass → planner, fail → verifier only). G9c-3 (can a deterministic judge replace the operator: pass → it judges the loop, fail → keep the operator).

**Measurements** have no bar. They produce a number another decision needs. G2 (how long does one world-model decision take on the Mac: 108 ms for W1, 1.8 s for W2; the number feeds the "can W2 be in the loop" decision but is not itself pass/fail). G11-A (latency of a surprise check: 12.78 ms). The chunk-stability statistic in the prep queue.

The rule applies differently. A gate without a decision should not run. A measurement without a *consumer* (a later decision that needs the number) should not run either, but its output is a number, not a verdict, and must be recorded as such. The most common corruption is retrofitting a bar onto a measurement after seeing the number: "12.78 ms, that's under 20, call it a pass". G11-A's protocol did have a bar (< 20 ms inside a 33 ms tick) written before the run, which is why it was legitimately a gate; the point is that the bar must exist before the number, whichever kind of experiment it is.

## 1.4 Worked cases from this repository

**The RoboJEPA 1B re-run, cancelled (2026-10-15).** Gate A had shown RoboJEPA's 1B model borderline on 30 clips. The proposed follow-up: run it on the full G1b protocol. Decision sentence: if it passes, RoboJEPA becomes the off-board world model; if it fails, nothing changes. But Gate B had already shown the encoder takes 526 ms for two views on the Orin, so a pass would *also* change nothing in the robot loop: an off-board model with 1 s round trip was not going to be used either way. X = Y. Cancelled; the protocol was amended to "optional zero-shot write-up, not a gate". Cost saved: ~$5 and an evening. Value of the write-up, had it been done: a sentence in a blog post.

**G7b and G8, never run.** The G7 protocol is a chain: G7a (outcome ranking, offline) → G7b (close-candidate rankability) → G8 (closed loop on the robot). Each later stage only runs if the earlier passes, because a fail at G7a already triggers "drop world-model selection", and G7b/G8 cannot change that. Writing the chain as a chain, with stop conditions, is the decision rule applied to a sequence.

**H0 (2026-10-08).** Decision: if VLA-JEPA's 8 samples vary enough, build the reranker; if not, drop reranking and move to failure prediction. They did not (1.08° spread; the mean of 8 captures 45 % of the oracle gain). The decision was taken the same day and the architecture changed. Total cost: an afternoon on the Orin, no GPU. This is what the rule buys: a no-go in one afternoon instead of three weeks of reranker engineering.

**The fifteen entries in NEXT_ANGLES (2026-10-22), as a violation.** Written in one evening, each with a "decision it changes" line, so nominally compliant. But several had decisions that were, on inspection, not decisions: §5 (world-model scaling curve: "a measurement, not a gate; pre-register the plot") feeds no decision in the current plan; §8 (lighting robustness) answers a question nobody was asking before the LEGO cell exists. They were included because they were *interesting* and because the file was being written quickly. The honest status of such entries is "reading list", not "queue". The lesson is that the decision sentence must be written *slowly*, one per evening, by the person who will act on it.

## 1.5 The alternatives, and why not

**"Explore first, decide later."** The argument: you do not know what matters until you look. True for the first week with a new system (the policy benchmarks of 09-29 were exploration, and rightly so). False after that: by week two the register already contained more results than could be acted on. Exploration without a decision is how a repo fills with results that contradict each other and nobody notices (G9b "running since 10-16" sat for a week).

**"Negative results are valuable, so run everything."** A negative result is valuable when it was a pre-registered test of a hypothesis someone would have acted on. A negative result from an experiment nobody would have acted on either way is a number with a date. The field is full of the second kind; they are not cited, because they changed nothing.

**"The paper is important; we should reproduce it."** Reproduction is a decision when the result would change what you build. TurboVLA's faithful reproduction (path A) is worth $2.15 because a pass changes the write-up and the plugin, and a fail upgrades fact 7 from "a property of my port" to "a property of the data". Reproducing H-JEPA's full hierarchy would not have been, because G6b had already shown the level-2 features fail on your trajectories' loops.

## 1.6 What the interviewer will ask

*"How do you prioritise experiments?"* The sentence. Then one example where it cancelled something (RoboJEPA 1B) and one where it ended a line in an afternoon (H0).

*"Isn't that too restrictive? Some of the best results are unplanned."* Agree that the first week on a new system is exploration, and say so. Then: after that, unplanned results come from reading failures (chapter 8), not from unplanned experiments. The TurboVLA fact 7 was unplanned; it came from six probes on a planned failure.

*"Give me an experiment you ran that you shouldn't have."* Be ready with one. G6a's done detector (T2, AUROC 1.00) was an easy contrast whose pass changed nothing because the negatives were at-rest frames. The time would have been better spent collecting real failures for G7.

## 1.7 Exercises

1. For every row in `EXPERIMENTS.md` §2, write the decision sentence in the margin. Mark the rows where X = Y. There are at least three. Write one paragraph on what you would have done with the saved time.
2. Take the eleven prep-queue entries. For each, rewrite its decision in your own words. Reject at least two. Commit the rejections with the reason.
3. Write the decision sentence for the robot day (G13a + G16-lite + G7a session). If any of the three blocks has X = Y, drop it from the day.
