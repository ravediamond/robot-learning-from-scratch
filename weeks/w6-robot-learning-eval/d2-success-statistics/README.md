# W6 D2: Success-rate statistics you can do on a whiteboard

Primer: `knowledge/07-robot-eval-statistics.md`. Book: chapter 7 (uncertainty). Coding list 17, 18.
Your register reports Wilson and bootstrap intervals everywhere: today you write them yourself.

## The four tools
**Wilson interval** for k successes in n trials (use it, not p̂ ± 1.96·se):
```
p = k/n;  d = 1 + z²/n;  centre = (p + z²/(2n)) / d;  half = z·sqrt(p(1−p)/n + z²/(4n²)) / d
```
**Episode bootstrap**: resample EPISODES with replacement B times, recompute the statistic, take the
2.5% and 97.5% percentiles. The episode is the unit of independence, never the frame.

**Paired comparisons**: run A and B from the same start states. Bootstrap the per-start difference;
for binary outcomes, McNemar's exact test uses only the discordant pairs (A ok / B fail = n01,
A fail / B ok = n10): under "no difference" each discordant pair is a coin flip:
`p = min(1, 2 · P(Bin(n01 + n10, 0.5) <= min(n01, n10)))`.

**How many trials** to tell p1 from p2 (two independent groups, two-sided α, power 1−β):
```
n per arm = ( z_{1-α/2} · sqrt(2 p̄ q̄) + z_{1-β} · sqrt(p1 q1 + p2 q2) )² / (p1 − p2)²,   p̄ = (p1+p2)/2
```
Use `statistics.NormalDist().inv_cdf` for the z values.

## Today's build (`w6d2_stats.py`)
`wilson`, `bootstrap_ci`, `paired_bootstrap_diff`, `mcnemar_exact`, `trials_needed`.

Run: `uv run pytest weeks/w6-robot-learning-eval/d2-success-statistics`
