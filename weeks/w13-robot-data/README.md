# Week 13: robot data engineering

Primer: `knowledge/14-robot-data-engineering.md`. Builds on W2 D4 (reading LeRobot data, gaps, idle starts, lag).

In robot learning the data is the model's ceiling, and it is collected by tired humans on hardware
that drifts. Research teams now hire for this explicitly ("Research Engineer – Data"): someone who can
say what is in 10,000 hours of teleoperation, what should not be, what to collect next, and prove that
a mixture change helped. This week builds that toolkit.

| Day | Lab | Review bug |
|---|---|---|
| d1 | dataset QA at scale: frozen cameras, dead joints, limits, robust outliers, duplicates | outliers by mean ± 3σ |
| d2 | curation: near-duplicates, removal ledger, group-aware splits, dataset fingerprint and card | random episode split leaks sessions |
| d3 | mixtures and scaling: temperature sampling, per-item weights, scaling-law fit, ablations with seeds | source weight not divided by its size |
| d4 | video and storage at scale: budgets, keyframe-aware sampling, worker sharding, shuffle buffers | every worker reads the same shards |
| d5 | the flywheel: intervention segments, failure mining, what to collect next + **design** | corrections labelled with the policy's actions |
