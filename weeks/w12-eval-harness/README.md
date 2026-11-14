# Week 12: the evaluation harness (your flagship engine)

Primer: `knowledge/13-evaluation-at-scale.md`. Builds on W6 (Wilson, McNemar, paired bootstrap, closed-loop eval).

W6 taught you to compare two policies on a toy. A real comparison of four policies on an SO-101 is
where most published robot results quietly go wrong: the session drifts (battery, lighting, the
operator learns), the operator knows which policy is running, 50 rollouts cannot see a 10-point gap,
"failure" means five different things, latency is measured on a laptop, and nobody can rerun the
exact checkpoint six months later.

This week you build the harness that fixes each of these. `flagship/` then turns it into the real
experiment: ACT vs SmolVLA vs VLA-JEPA vs your own head on your arm, 50+ rollouts each.

| Day | Lab | Review bug |
|---|---|---|
| d1 | protocol: counterbalanced blocked schedule, blinding, record validation, drift | policies run one after the other |
| d2 | several policies: McNemar per pair, Holm, rank probabilities, minimum detectable difference | bootstrap breaks the pairing |
| d3 | failure taxonomy: rater agreement, per-cause rates with CIs, stage funnel | rates computed over failures only |
| d4 | latency on the target: percentiles, deadlines, sync vs async chunk execution | p99 from mean + std |
| d5 | reproducibility and the report: fingerprints, run manifests, results table + **design** | checkpoint hashed through `str()` |
