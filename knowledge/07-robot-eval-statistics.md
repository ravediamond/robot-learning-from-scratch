# 07. Evaluating robot policies: statistics that survive an interview

This is your strongest area on paper (agentic + eval background, the nine-step method, the
register). The goal here: be able to DO the numbers on a whiteboard, not just name them.
Book depth: chapters 1–9, especially 7 (uncertainty). Coding list items 16–19.

## The method in nine steps (your own, from the book)
1. Start from the decision the result will change. 2. Cheapest baseline first.
3. Pre-register the bar (before seeing results). 4. One factor per ablation.
5. Verify the instrument (does the measurement measure what you think?).
6. Episode-level bootstrap for uncertainty. 7. Read the failures, not just the rate.
8. Three verdicts: go / no-go / inconclusive. 9. Log everything in the register.

## Success rates with small n

k successes out of n trials. Point estimate p̂ = k/n. With n = 10–30, the uncertainty is
huge, and the normal approximation (p̂ ± 1.96√(p̂(1−p̂)/n)) breaks near 0 and 1.

**Wilson interval** (use this):
```
center = (p̂ + z²/(2n)) / (1 + z²/n)
half   = z √(p̂(1−p̂)/n + z²/(4n²)) / (1 + z²/n)        z = 1.96
```
7/10 → about [0.40, 0.89]. 9/10 → about [0.60, 0.98]. So "9/10 vs 7/10" proves nothing.

**How many trials?** To distinguish 60% from 80% with reasonable power you need on the order
of 80+ trials per arm unpaired. That is why paired designs matter.

## The unit of independence is the episode

Frames within an episode are correlated; treating them as independent shrinks CIs by
10×+ and invents significance. Bootstrap over episodes:
```python
for b in range(B):
    idx = rng.integers(0, n_episodes, n_episodes)    # resample EPISODES with replacement
    stat[b] = metric(episodes[idx])
ci = np.percentile(stat, [2.5, 97.5])
```

## Paired comparisons

Run policy A and B from the SAME initial conditions (same object layout, same seed). Then
analyze differences per start state: the variation due to layout difficulty cancels.
- Binary outcomes, paired: McNemar's test uses only the discordant pairs (A succeeds/B fails
  vs the reverse).
- Bootstrap the paired difference by resampling start states.

## Monitors and detectors: AUROC and beyond

AUROC = probability that a random failure gets a higher score than a random success.
Rank-based computation (Mann–Whitney). Always report with an episode-bootstrap CI, then
the operating point you would deploy (TPR at FPR = 5%) and the lead time.

## Multiple comparisons

Try 20 variants, one will look significant at p < 0.05 by chance. Pre-register the primary
comparison; treat the rest as exploratory; or correct (Bonferroni: α/m). Picking the best
checkpoint on the same trials you report = the same sin.

## Questions

1. Compute the Wilson interval for 7/10 on the whiteboard.
<details><summary>Answer</summary>p̂ = 0.7, n = 10, z² = 3.84. Denominator 1 + 0.384 = 1.384. Center = (0.7 + 0.192)/1.384 = 0.645. Inside sqrt: 0.21/10 + 3.84/400 = 0.021 + 0.0096 = 0.0306 → √ = 0.175; × 1.96 = 0.343; / 1.384 = 0.248. Interval ≈ [0.40, 0.89].</details>

2. Policy A 14/20, policy B 10/20, same start states. Is A better?
<details><summary>Answer</summary>Unpaired Wilson intervals overlap heavily ([0.48, 0.85] vs [0.30, 0.70]). Paired: look at discordant pairs. If A-only successes = 5 and B-only = 1, McNemar exact p ≈ 0.22: not significant. You need more trials, or a pre-registered sequential design. Say "inconclusive", with the numbers.</details>

3. Why bootstrap episodes, not frames?
<details><summary>Answer</summary>Frames within an episode are strongly correlated (same scene, same failure). Resampling frames treats 100 correlated frames as 100 independent observations, giving intervals far too narrow. The episode is the independent unit of the experiment.</details>

4. What does "pre-registration" protect against, concretely, in a robot lab?
<details><summary>Answer</summary>Moving the goalposts after seeing results: choosing the metric, the threshold, the subset of tasks, the checkpoint, or the number of trials that makes your method look good. Writing the bar and protocol down (commit hash, date) before running makes the result a test instead of a story.</details>

5. Your monitor has AUROC 0.85 [0.74, 0.93]. Is it useful?
<details><summary>Answer</summary>Depends on the baseline and the operating point. If elapsed time already gives 0.93 (your length-null result), it adds nothing. If it beats the baseline on paired episodes, check TPR at an acceptable FPR (false alarms cost resets) and lead time (an alarm at the moment of failure is useless for recovery). AUROC alone is not a deployment decision.</details>

6. Interviewer: "You claim 9/10 success. How confident are you it is better than 50%?"
<details><summary>Answer</summary>Wilson 95% interval for 9/10 is about [0.60, 0.98], so above 50% at 95%, but the true rate could be 60%. And that is only for this layout set, this lighting, this day: a claim about the policy needs varied start states and repeated sessions. I would say "9/10 on 10 fixed layouts, CI 60–98%".</details>

7. How would you design an eval for a long-horizon task (10 subtasks) on one arm with limited time?
<details><summary>Answer</summary>Report per-subtask success and the progress curve (how far each episode gets), not only full-task success (which is ~0 with small n and hides where it breaks). Fix a set of start states, run paired comparisons, use resets you can script, bootstrap over episodes, pre-register the primary metric (e.g. mean subtasks completed), and log failure types per subtask.</details>
