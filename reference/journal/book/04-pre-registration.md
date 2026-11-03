# Chapter 4. Pre-registration

## 4.1 The principle

Before the first number exists, write into a file: the question, the data and split, the method, the baselines, the measurement checks, the metric and its statistics, the pass number, the fail number, the decision each triggers, and the cost. Commit it. The commit hash is the bar's birth certificate. Everything decided after that commit is labelled *exploratory* and cannot change the verdict.

## 4.2 Why: the garden of forking paths

Any analysis has dozens of small choices: which episodes count, which frames, which metric, which threshold, whether to drop that one weird run, whether the bar is ≥ or >. Each choice is individually defensible. Made *after* seeing the data, they are made in the direction of the result you hoped for, usually without noticing. With twenty binary choices there are a million analyses; one of them passes. Gelman and Loken named this the garden of forking paths: you walk one path, it looks like the only path, and you report it as if you had planned it.

Pre-registration closes the garden by making the choices before the data can influence them. It does not require that you were right about the choices, only that they were made blind. A pre-registered fail is a fail of the method *as specified*; it can be followed by exploratory analysis that suggests a better specification, which then gets its own pre-registered test.

Why it matters more for a one-person lab than for a team: a team has colleagues who will ask "did you decide that before or after". Alone, nobody asks. The commit is the colleague.

## 4.3 What a frozen protocol contains

The template used in the prep queue, with what each section is for:

| Section | What it fixes | Forking path it closes |
|---|---|---|
| Question | one sentence | scope creep after the fact ("we were really asking about…") |
| Decision rule | pass → X, fail → Y | retrofitting a decision to a result |
| Data and split | dataset ids, pinned commits, episode-level split, seeds | choosing the split that passes; frame leakage across train/test |
| Method under test | hyper-parameters, reference implementation, parity check | "we meant the tuned version" |
| Baselines (computed first) | each named, with its computation | dropping the baseline that wins |
| Measurement checks | strict load, parts-must-sum, parity tolerance, smoke run | measuring the wrong thing (chapter 6) |
| Metric and statistics | metric, bootstrap by episode, paired CI, sample size | switching metric; frame-level CIs |
| Bar | PASS: number. FAIL: number. | moving the threshold |
| Cost | $ and robot days | "just one more run" |
| Outputs | result path, register row, draft sections | losing the result |

A protocol with any of these blank is not frozen. The prep queue's first version had the bar blank on purpose so the experimenter would write it; the second version filled it so Claude Code could run; in neither case could a run start on a blank bar.

## 4.4 The rules this repository follows

1. **Protocol committed before the first number.** The register row cites the commit (e.g. `37497f0` for W1-A, `2bcd2ff` for W1-C). Claude Code's instruction is literally "commit the protocol with message '<NAME> protocol frozen' before computing any number."
2. **Amendments are committed before the number they affect.** The D-JEPA protocol was amended twice (once when the paper turned out to describe an operator, not a loss; once to run on frozen W1); each amendment preceded the run it governed. The G7 cube amendment was committed with its flags before any cube rollout exists.
3. **Everything after is exploratory, and says so.** G1's "Deviation note" and the exploratory top-2 ceiling in the D-JEPA write-up are labelled. They are allowed to *suggest* (regret as the next metric; the candidate set as the real limit) and not to *decide*.
4. **Rows are never rewritten.** A correction is a new row (G11-A re-check, G12 re-check, G9a-AB complete, G9b never-ran).
5. **Chat-only facts are provisional** until a file records them; the register marks them *chat*.

## 4.5 The hard case: adopting a narrow miss

G9c-3 scored 94.3 % against a pre-registered 95 % bar and was adopted as the primary judge. Was that a violation?

The argument for adoption: every one of its errors was a success whose recording ran past the operator's reset (so the final frame showed separated bowls); its false-positive count was 0, and zero false positives was the property the training loop needed; the 95 % accuracy bar was the wrong bar for the decision, written before the failure mode was understood.

The argument against: this is exactly the reasoning the garden of forking paths warns about. "The bar was wrong" is what everyone says after a miss.

The resolution this repo uses: a narrow miss may be adopted **only if** (a) the adoption is written as a miss in the register ("narrow fail / operational pass"), (b) the property that justified it was *measured and reported before the decision* (the 0-FP count was in the G9c report, not added later), and (c) the bar is corrected for the next gate (subsequent judge bars in the prep queue say FP = 0 first, accuracy second). What is not allowed: adopting and calling it a pass; adopting without changing the next bar; adopting on a property that was computed to justify the adoption.

The honest interview answer: "I adopted a judge that missed its bar by 0.7 points, because the bar was accuracy and the decision needed zero false positives, which it had. I wrote it down as a miss and changed the bar for the next one. The lesson was that I had pre-registered the wrong metric, and pre-registration is what let me see that."

## 4.5a The second hard case: a failed bar read as not proven

HWM (12b.8b, 2026-10-28) failed two of its three bars and was written *not proven* by the owner, because ~0.6 h of one task cannot test a crossover the paper found at ~130 h. The bars were kept and printed; only the verdict word changed. It is defensible and it is still the weakest verdict in the register, because the reading was applied after the result. The fix is one line in the protocol, before the run:

> *Verdict map.* Pass: all bars met. Fail: H1 or H2 missed **and** the data is within 10× of the paper's scale. Not proven: H1 or H2 missed with the data more than 10× smaller, or the result between the pass and fail lines.

Written that way, the same numbers give the same verdict without an owner's decision, and the interviewer's "so you overruled your own bar?" has the answer "no, the protocol said what a miss at this scale means". The rule from here on: every protocol carries a verdict map.

## 4.6 Statistics of a bar: how to pick the number

A bar is a number that separates "act" from "don't act". Three ways to set it, in order of preference:

**From the decision's cost structure.** A judge in a training loop: a false positive poisons training, a false negative wastes a demo; FP = 0 and FN ≤ some rate. A monitor that stops the arm: each false intervention costs a trial and operator trust; ≤ 1/20 is "one annoyance per session". A chunk method: the boundary jump must fall by enough to matter to the servos (≥ 50 %).

**From the baseline plus a margin the sample can resolve.** "Beat episode length by ≥ 0.05 AUROC" is only meaningful if 69 episodes can resolve 0.05. Roughly: the SE of an AUROC at ~0.85 with n₊ ≈ 25 and n₋ ≈ 44 is about 0.05 (0.053 by Hanley-McNeil), so a difference of 0.05 is about one standard error, not detectable; 0.10 is safer. The prep queue uses ≥ 0.10 for that reason. Chapter 7 gives the arithmetic.

**From the paper's own margin.** If the paper claims +8 points and you have a quarter of its data, requiring +8 is unfair and requiring +0 is meaningless; require the paper's effect at your noise level, with the CI excluding 0.

What is not a way to pick the bar: a round number that sounds good (0.8, 95 %). G1b's 0.8 top-1 was such a number; the p^7 arithmetic showed it was unreachable by any method on this data, so the gate could only fail. That is a design error in the bar, caught in the diagnosis. The fix was the right metric (regret), not a lower bar.

## 4.7 Pre-registration of robot days

Robot days are the most expensive experiments and the easiest to improvise. The runbook (`ROBOT_DAY_RUNBOOK_G13A_G16LITE_G7A.md`) is the pre-registration: the order, the trial counts per arm, the layouts (sealed list, placed by the experimenter, v1.5 §11), the failure modes to stage, the bars (already frozen in G13a, G16-lite, G7), and the rule that if the day runs long the G7a block moves, not the bars. Decisions made at the table ("let's do five more of the ones that worked") are exploratory and recorded as such.

## 4.8 What the interviewer will ask

*"Do you pre-register? Isn't that academic?"* "Yes, in a committed protocol file with the bar as a number. It took three days of results to see why: without it I had a judge that 'passed' on a partial set and failed on the complete one, and I could not tell which analysis I'd have believed if I hadn't written it down first."

*"What happens when the pre-registered bar turns out to be wrong?"* G9c-3. The three conditions. "I'd rather adopt a documented miss than quietly move a threshold."

*"How do you choose the bar?"* 4.6, in order. Then the G1b lesson: a bar can be unreachable by construction, and the fix is the metric, not the number.

## 4.9 Exercises

1. Read `G6A_PROTOCOL.md`. List every analytic choice it fixes. Then list five it does not fix that could have been used to rescue a pass. Propose the amendment that closes them.
2. For G9c-3, write the bar that should have been written (FP, FN, accuracy, in that order), with the decision each outcome triggers.
3. For each prep-queue bar, say which of the three ways in 4.6 it was set by, and whether 69 episodes can resolve it. Amend at least one.
4. Write the pre-registration for the robot day in one page: arms, n per arm, layouts, bars, stop rules, what moves if time runs out.
