# W6 D3: Failure detection: AUROC, operating points, conformal thresholds

Book: 7.5 (AUROC and its standard error), 13.2 (monitors). Primer 06 and 07. Your paper:
`reference/journal/papers/failure-monitors/` (the length null).

## A monitor is a score + a threshold
Score per episode (or per step): higher = more likely to fail. Label: 1 = failure.
- **AUROC** = P(random failure scores higher than random success), ties count ½. Rank form
  (Mann–Whitney): rank all scores (ties get their AVERAGE rank), then
  `AUROC = (sum of failure ranks − n1(n1+1)/2) / (n1 · n0)`.
- **CI**: bootstrap over episodes (resamples with a single class are skipped).
- **Operating point**: you deploy a threshold, not an AUROC. TPR at FPR = 5%: threshold = the
  95th percentile of the SUCCESS scores; fraction of failures above it.
- **Conformal threshold**: from n calibration successes, take the k-th smallest score with
  k = ⌈(n+1)(1−α)⌉. On a new success episode, P(score > threshold) ≤ α (exchangeability).
  A guaranteed false-alarm rate with no model assumptions.

**The cheapest baseline: episode length / elapsed time.** Failures tend to run long. Your paper found
this null hard to beat (within-policy AUROC CI lower bound 0.93 on ARMNetBench). Every monitor must
beat it on the same episodes, paired.

## Today's build (`w6d3_monitors.py`)
`auroc`, `auroc_ci`, `tpr_at_fpr`, `conformal_threshold`.
Then on your Mac: `uv run python length_null_real.py` computes the length null on your own
`ravediamond/so101-bowls-policy-rollouts` labels with YOUR functions.

Run: `uv run pytest weeks/w6-robot-learning-eval/d3-failure-detection`
