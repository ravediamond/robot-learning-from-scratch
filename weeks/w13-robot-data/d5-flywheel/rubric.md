# Rubric

Score each line 0 / 1 / 2.

| # | Criterion | Senior signal (1) | Staff signal (2) |
|---|---|---|---|
| 1 | Requirements | volumes restated | consumers (research, eval, ops) and their latency needs, retention per data class, residency, non-goals |
| 2 | Estimation | hours per day | episodes/day, compressed TB/day and /year, raw vs compressed decision, egress and storage tiers costed |
| 3 | Edge & ingestion | "upload to S3" | local buffering with integrity checks, resumable uploads, episode as the atomic unit with metadata (station, operator, calibration hash, firmware), idempotent ingestion |
| 4 | Quality | "validate data" | automated checks at ingestion (W13 D1 list), per-station dashboards and alerts within minutes, quarantine with reasons, live checks at recording time |
| 5 | Catalogue & versioning | "dataset versions" | per-episode content hashes, dataset fingerprints, removal ledger, split definitions by group, dataset cards, reproduce any training set from (fingerprint, ledger, split) |
| 6 | Training access | "data loader" | sharded streaming format, re-encoded training copies (GOP, resolution), worker sharding, decode budget computed, mixture sampling with explicit weights |
| 7 | Flywheel | "retrain on new data" | interventions as corrections with executed actions, failure mining queue, labelling tools and κ, targeted collection planning, measured per iteration |
| 8 | Privacy & residency | "GDPR" | EU-only buckets per site class, face blurring at the edge or ingestion, access control per dataset, deletion propagates (ledger + derived copies), audit logs |
| 9 | Operations | monitoring | ingestion lag, QA flag rate per station, storage growth, loader throughput, cost per hour of data; on-call for the pipeline |
| 10 | Judgment | builds everything | v1 = ingestion + QA + catalogue on managed object storage and LeRobot format; buy labelling tools; defer custom formats; what changes at 10× robots |

0–9 senior or below · 10–14 senior+ · 15–17 staff · 18–20 strong staff.
