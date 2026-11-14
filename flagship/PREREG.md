# Pre-registration (fill BEFORE any rollout, commit, and never edit: deviations go at the bottom)

Commit hash of this file at registration: (write it here after committing, in a second commit)

## Question
On the SO-101 cube-to-bowl task with the same training data, how do ACT, SmolVLA, VLA-JEPA and my head
compare in success rate, failure causes and on-device latency?

## Hypotheses (one sentence each, with the direction)
- H1: my head's success rate is higher than ACT's.
- H2: VLA-JEPA's success rate is higher than SmolVLA's.
- H3: SmolVLA fails more often on `grasp` than ACT.
- H4: every policy meets a 33 ms deadline-miss rate below 5% in its declared execution mode on the Orin.

## Primary analysis
- Outcome: success (binary) per rollout, 50 analysed rollouts per policy, paired by start configuration.
- Tests: exact McNemar for the pairs in H1 and H2; Holm across all 6 pairs (the others reported as exploratory).
- α = 0.05 family-wise. Report Wilson 95% CIs per policy and paired bootstrap CIs per pair.
- Minimum detectable difference at n = 50 around 50% success: about 27 points (unpaired approximation).
  Differences below that are reported as "not resolved", not as "no difference".

## Secondary analyses
- Failure causes per policy as rates over all rollouts with Wilson CIs; stage funnel; κ of the second rater.
- Rank probabilities (bootstrap over start configurations).
- Latency p50/p95/p99, deadline misses at 30 Hz, async starvation check for the declared chunk settings.

## Exclusions
Only `aborted` rollouts as defined in `PROTOCOL.md`; warm-up rollouts. Everything else is analysed.

## What would change my mind
- If H1 fails with a CI that excludes +10 points, my head is not better on this task: I report it as such.

## Deviations (append only, dated, with the reason)
-
