# 13. Evaluation at scale: protocols, comparisons, failures, latency, reproducibility

Week 12 and `flagship/`. Primer 07 covers the statistics of one comparison; this one covers running
an evaluation program that a research team can trust.

## 1. Protocol
Fixed start configurations (paired comparisons). Blocks of one start with every policy once, order
rotating (counterbalanced): drift hits all policies equally. Operator blind to the policy (aliases).
Written success criteria, timeouts, reset procedure, intervention rules. Records validated at write
time with a fixed schema. Warm-up rollouts declared in advance. A frozen reference policy in every
session to detect setup drift (A/A).

## 2. Several policies
Exact McNemar per pair on paired outcomes; **Holm** across pairs (family-wise error); rank probabilities
by resampling start configurations jointly; **minimum detectable difference** stated before results
(about 27 points at 50 rollouts per arm around 50% success). No optional stopping: fixed n or a proper
sequential design. Training seeds are a separate variance source: report per seed.

## 3. Failures
Closed taxonomy with observable definitions, one primary tag per failure, double labelling of a sample
with Cohen's κ (≥ 0.6 to publish per-cause rates). Per-cause rates over ALL rollouts with Wilson CIs,
never shares of failures. Stage funnel (reach → grasp → lift → place) shows where each policy loses.

## 4. Latency on the target
On the robot's computer, fixed power mode, warm, GPU synchronised, real preprocessing. Report p50/p95/
p99/max (nearest rank), deadline misses. Chunked control: sync idles `L/(L + e·T)`; async needs
`L ≤ e·T` and `ceil(L/T) + e ≤ H`; count starved steps with measured latencies. The execution mode is
part of the method.

## 5. Reproducibility
Weight fingerprints (SHA-256 over names, dtypes, shapes, bytes), run manifests (code, weights, dataset,
config hash, execution settings, hardware, packages, protocol version), manifest diffs, generated
tables. Pre-registration committed before the first rollout; deviations appended, never edited.

## 6. Programs, not experiments
Tiers: offline metrics (sanity) → sim / world-model screening validated by rank correlation against
real outcomes → real robots decide. Budget robot-hours explicitly. Promotion rules pre-registered
(non-inferiority per task, no safety regressions). Station drift monitored with the reference policy.

## Questions
1. Four policies, 50 rollouts each, best is 8 points above second. What do you say?
<details><summary>Answer</summary>Not resolved: the MDE is ~27 points; give per-policy Wilson CIs, the Holm-adjusted pairwise p, rank probabilities, and what n would resolve it.</details>

2. Why counterbalance the order inside blocks?
<details><summary>Answer</summary>So that no policy is systematically first after a reset or last before a break; position effects cancel across blocks.</details>

3. Share of failures vs rate per rollout?
<details><summary>Answer</summary>Shares are relative to each policy's own failures and cannot be compared across policies; rates per rollout can, with CIs.</details>

4. What goes in a run manifest that people forget?
<details><summary>Answer</summary>Execution settings (chunk, actions executed, sync/async, flow steps), camera settings, arm calibration hash, power mode.</details>

5. When can a world model be used as an evaluator?
<details><summary>Answer</summary>When its policy rankings were shown to agree with real outcomes (rank correlation, MMRV) on the same tasks and a comparable set of policies, and only for pruning, not for the final decision.</details>
