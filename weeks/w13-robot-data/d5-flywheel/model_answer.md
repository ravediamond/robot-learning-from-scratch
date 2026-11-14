# Model answer: data platform for a robot fleet

**Numbers.** 100 arms × 8 h = 800 robot-hours/day; at ~1 min per episode ≈ 48,000 episodes/day. Video:
3 cameras × ~1 Mbit/s ≈ 1.35 GB per robot-hour → ~1.1 TB/day compressed, ~400 TB/year (raw would be
~240 TB/day: not kept). Low-dimensional data: a few GB/day.

**Edge.** Each station records in LeRobot format to local SSD, an episode at a time, with metadata:
station, site, operator, task, calibration file hash, firmware, policy fingerprint if autonomous,
intervention flags, executed and proposed actions. Live checks at recording time (frozen camera,
dropped frames, limit hits) warn the operator immediately. A daemon uploads finished episodes with
checksums and resumable transfers; the episode id is a content hash, so retries are idempotent.

**Ingestion and storage.** Object storage per residency class (EU-only buckets for customer sites that
require it). Ingestion validates schema and checksums, runs the QA checks (W13 D1), computes content
hashes, writes the episode to the catalogue (a database: episode → files, metadata, QA results,
quarantine reasons) and to the hot tier. Faces blurred at ingestion for customer sites, originals not
kept. Tiers: hot (last 90 days, training copies), warm (all, original encoding), cold archive.

**Quality monitoring.** Per-station dashboards of flag rates, frame drops, calibration drift; alerts when a
station's flag rate jumps (within the hour). Quarantine, never delete; the ledger records every decision.

**Catalogue and versioning.** A dataset = a query over the catalogue (tasks, sites, dates, QA filters) +
ledger state + split definition by group (session/station/operator), frozen into a manifest of episode
hashes with a fingerprint and a dataset card. Training runs record the fingerprint; reproducing run 412
= loading its manifest.

**Training access.** A materialisation job writes training shards (re-encoded with short GOPs at training
resolution, parquet for low-dimensional data) for a dataset version; streaming loaders assign shards by
(rank, worker), shuffle buffers, explicit mixture weights by source. Decode budget computed per run
(frames/s × decoded frames per sample ÷ per-core throughput) and the loader CPU sized accordingly.

**Flywheel.** Autonomous episodes flow through the same path; intervention segments become correction
data labelled with executed actions; failure-detector scores rank a review queue; reviewers tag
failures with the taxonomy (double-labelling sample, κ); weekly collection plans allocate demos to
the conditions that fail; each iteration is evaluated with the blocked protocol.

**Privacy.** Residency by bucket and by training job location; access per dataset role; deletion
requests propagate through the ledger to training shards (rebuild) and are logged; models trained on
deleted data are flagged for the next retrain.

**Not in v1.** Custom storage formats, automatic failure tagging, cross-site active learning. Buy a
labelling tool. Start with one site, then add the others.
