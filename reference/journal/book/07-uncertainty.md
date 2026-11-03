# Chapter 7. Uncertainty: episodes, pairs, small n

## 7.1 The principle

Every number in a result table carries an interval, the interval is computed over the right unit of independence, comparisons between methods use paired intervals of the difference, and the sample size is stated next to every number. A result without these is an anecdote with decimals.

This chapter derives the tools rather than naming them, because an interviewer at a company that runs real-robot evaluations will ask you to justify them, and "I used bootstrap" is not a justification.

## 7.2 The unit of independence is the episode

An episode of 600 frames at 30 Hz is not 600 observations. The frames share a layout, a lighting condition, a policy sample, an operator state, a camera exposure. Two frames one second apart are nearly identical for most purposes. If you compute a confidence interval treating frames as independent, the standard error is √(600) ≈ 24 times too small per episode, and a method with no real effect will appear significant.

The rule: the unit is the thing that was independently drawn. In teleop data, the episode (each is a new reset). In policy rollouts, the episode. In the LEGO cell, arguably the layout (several picks share a layout). In a robot day, the trial, and within a day trials are not fully independent either (operator fatigue, servo heating), which is why layouts are sealed and arms interleaved.

**The episode bootstrap.** To get a 95 % interval on a metric M computed over E episodes:

```
for b in 1..2000:
    S_b = sample E episodes from the E, with replacement
    M_b = M computed on all frames of the episodes in S_b
interval = [percentile(M_b, 2.5), percentile(M_b, 97.5)]
```

Why 2 000: the percentile estimates stabilise to ~0.01 by then; 200 is noticeably noisy, 20 000 changes nothing. Why percentile and not ±1.96·SE: metrics like AUROC near 1 are skewed and bounded; the percentile interval respects the bound.

What it does not fix: it assumes the E episodes are a sample from the population you care about. 69 rollouts from five policies on one task in one room are a sample from that room and those policies. The interval says nothing about a different task or a stronger policy (chapter 12 on coverage shift).

## 7.3 Paired comparisons

The question is almost never "what is A's AUROC" but "is A better than B". Two separate intervals do not answer it:

> A: 0.86 [0.80, 0.91]. B: 0.82 [0.75, 0.88].

Overlapping intervals are consistent with A ≫ B *and* with A ≈ B, because A and B are evaluated on the same episodes and their errors are correlated: when a hard episode is resampled out, both scores rise together. The right object is the distribution of the *difference* under the *same* resamples:

```
for b in 1..2000:
    S_b = same resample for both
    D_b = M_A(S_b) − M_B(S_b)
interval on D
```

If the interval on D excludes 0, A beats B at that confidence; if it includes 0, the data cannot tell. The G9b re-analysis reported "Robometer added to length: +0.008, CI includes 0"; that sentence is the whole result. The prep-queue bars are written as "≥ +0.10 over episode length with the paired CI excluding 0" for this reason.

Interviewers will test this with "both methods had overlapping error bars, so there's no difference, right?" The answer is "overlapping marginal intervals say nothing about the paired difference; I'd want the CI on A − B."

## 7.4 Success rates with small n

Robot trials are expensive, so n is small, and the interval on a proportion at small n is wide in ways that intuition gets wrong.

**Wilson interval** for k successes in n trials (preferred over the naive p ± 1.96√(p(1−p)/n), which breaks near 0 and 1 and at small n):

centre = (p + z²/2n) / (1 + z²/n), half-width = z·√(p(1−p)/n + z²/4n²) / (1 + z²/n), with z = 1.96.

Worked values (95 %):

| k/n | p | Wilson interval |
|---|---|---|
| 9/10 | 0.90 | [0.60, 0.98] |
| 8/10 | 0.80 | [0.49, 0.94] |
| 10/10 | 1.00 | [0.72, 1.00] |
| 6/6 | 1.00 | [0.61, 1.00] |
| 2/6 | 0.33 | [0.10, 0.70] |
| 18/20 | 0.90 | [0.70, 0.97] |
| 54/60 | 0.90 | [0.80, 0.95] |
| 180/200 | 0.90 | [0.85, 0.93] |

So: 9/10 vs 8/10 is one trial, and the intervals span 40 points each. The 10-16 multi-task gate's "single objects 6/6, named cube among two 2/6" is [0.61, 1.00] vs [0.10, 0.70]: a real difference, but each number alone is nearly uninformative. The register's policy table (ACT 10/10, SmolVLA 9/10, VLA-JEPA 9/10) cannot rank them, which is why the text says "ceiling; decide on the hardest primitive".

**Clopper-Pearson** is the exact (conservative) alternative; Wilson is close enough and narrower. Use either; say which.

**Power: how many trials to see a difference.** To distinguish 80 % from 90 % success with a two-sided test at α = 0.05 and 80 % power, you need roughly n ≈ 200 per arm. To distinguish 60 % from 90 %, roughly n ≈ 32 per arm. To distinguish 50 % from 90 %, roughly n ≈ 20. The rule of thumb for planning: with 20 trials per arm you can detect gaps of ~30 points; with 50, ~20 points; with 200, ~10 points. The prep queue says "n = 6 per cell cannot resolve less than ~30 points" and "with n = 20 per arm, gaps under ~30 points are noise" for this reason, and the RESTA / quality-aware bars are written on *pooled* success over 21 tasks (n ≈ 126) so a 15-point gap is within reach.

Where this leaves robot days: 207 trials in a day is a lot of trials, and it is enough for ~3 two-arm comparisons at the 30-point level, or one at the 20-point level. Plan the day around that, not around the number of ideas.

## 7.5 AUROC: what it is and its standard error

AUROC is the probability that a randomly chosen positive scores higher than a randomly chosen negative. 0.5 is chance; 1.0 is perfect separation; it is threshold-free, which is why it is comparable across papers and why it hides the threshold decision that matters in practice (a monitor at AUROC 0.93 still intervened on 7/42 successes at the threshold that caught 6/8 failures).

Its standard error depends on both class counts. The Hanley-McNeil approximation at AUROC ≈ 0.85 with n₊ = 25, n₋ = 44 gives SE ≈ 0.053; at n₊ = 8, n₋ = 42 (the REAP primary set) SE ≈ 0.089 (corrected 2026-10-24 from 0.045 and 0.08; arithmetic in appendix S.7). So on the primary set a difference of 0.05 is invisible and 0.15 is borderline. Say this in the write-up; it is why "AUROC 0.86 vs 0.93" between W1 surprise and the risk head is not a ranking.

## 7.6 Multiple comparisons

Over three weeks, several dozen gates ran on the same 69 rollouts and 182 demos. Some will pass by chance. The usual corrections (Bonferroni, Holm) are for a family of tests of one hypothesis; here the gates ask different questions, and correcting across them would make every bar unreachable.

The defences used instead:

- **Pre-registration** (chapter 4): the choice of what to test was not made after seeing which tests passed.
- **Replication on fresh data**: G10's thresholds (fit on seen rollouts) were re-tested on 50 fresh rollouts (W1-A) and did not hold. That is the multiple-comparisons correction that matters: a pass must survive data it was not fit on.
- **Mechanism** (chapter 8): a pass with a mechanism (prefix overwrite reduces the jump because the first action now matches the arm) is less likely to be chance than a pass without one.
- **Reporting the count**: the register lists every gate, including the fails, so a reader can see that 1 pass in 10 attempts at the same question is what chance would give.

## 7.7 Reporting rules

Every table: n per row; interval per number; paired CI for every comparison claimed; the primary set and secondary set separated when policies mix; the sample's provenance (which rollouts, which commit).

Every sentence with a number: say what the number cannot tell. "AUROC 0.93 (n₊ = 8, n₋ = 42; SE ≈ 0.064), so differences below ~0.15 are not resolved."

Never: "significant" without the test; "trend" for a CI that includes zero; "9/10 vs 8/10" as a difference.

## 7.8 What the interviewer will ask

*"How many trials do you run per condition, and why?"* The power table. "Twenty per arm resolves thirty points; I plan robot days so each comparison I care about has at least that, and I say in the write-up what the n cannot resolve."

*"Your AUROC went from 0.86 to 0.93. Is that real?"* "Not at n₊ = 8; the SE is about 0.08 and the paired CI on the difference includes zero. What is real is that neither clears the false-intervention budget."

*"How do you bootstrap robot data?"* Over episodes, 2 000 resamples, paired for comparisons, percentile interval. And the caveat about what population the episodes represent.

## 7.9 Exercises

1. Implement Wilson and Clopper-Pearson; reproduce the 7.4 table; add 23/25 and 4/44 (the REAP secondary set).
2. Implement the paired episode bootstrap and run it on the C1 probe outputs (imagined vs current vs policy identity). Report the three paired differences with CIs.
3. For the robot-day runbook's 207 trials, write out which comparisons it can resolve at the 30-point level and which it cannot. Propose a cut to 150 trials that loses nothing you would act on.
4. Derive the Hanley-McNeil SE for AUROC = 0.85 at (8, 42) and (25, 44). Show the arithmetic.
