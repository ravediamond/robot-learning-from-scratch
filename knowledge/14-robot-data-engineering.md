# 14. Robot data engineering

Week 13. The "Research Engineer – Data" layer: what is in the data, what should not be, how it is
versioned, mixed, stored, fed to training, and refreshed from deployment.

## 1. Quality at ingestion
Checks per episode: schema and timestamps (W2 D4), frozen cameras (longest run of identical frame
hashes), dead joints (range ≈ 0 while others move), limit hits, calibration consistency, idle head/tail,
action–state lag, length outliers by **median/MAD** (mean ± σ masks outliers), duplicates by content
hash. Quarantine with reasons, never delete; per-station flag rates as a monitoring signal.

## 2. Curation and splits
Near-duplicates by resampled trajectories (fixed length, RMS or DTW). Removal **ledger** (append only,
reversible). Splits by **group** (session, operator, station, object set), never by random episode.
Dataset **fingerprint** = hash of sorted per-episode content hashes. Dataset cards.

## 3. Mixtures and scaling
Temperature sampling `p_i ∝ n_i^{1/T}`; per-item weight `p_source / n_items`; epochs per source
`B·p/n` (repetition risk for small, up-weighted sources). Scaling curves `err ≈ a·N^{−b}` on nested
subsets give the marginal value of more data of one kind. Ablations paired by seed with intervals.

## 4. Video and storage
Raw 640×480×3 at 30 fps ≈ 100 GB per camera-hour; compressed ~0.5 GB. Random access costs
(GOP+1)/2 decoded frames: sample contiguous windows, re-encode training copies with short GOPs,
decode on GPU, size loader CPUs. Streaming: shards assigned by `rank·workers + worker`, shuffle buffers
much larger than an episode, resumable loader state. LeRobot v3: parquet tables + MP4 per camera with
many episodes per file, metadata with offsets and normalisation stats.

## 5. The flywheel
Interventions → corrections labelled with EXECUTED actions; failure mining by detector score and
uncertainty; collection planning by failure rate plus uncertainty (largest remainder); each iteration
evaluated with the blocked protocol and its cost in human hours reported.

## Questions
1. Why median/MAD for outliers?
<details><summary>Answer</summary>Breakdown point 50%: outliers cannot move the centre or the scale, so they cannot hide each other (masking).</details>

2. What leaks in a random episode split?
<details><summary>Answer</summary>Session-level correlations: lighting, objects, operator habits, near-duplicate demonstrations. Validation then measures memorisation of sessions.</details>

3. Your source is 100 h next to a 1,000 h public set. How do you sample?
<details><summary>Answer</summary>Explicit mixture weights (temperature or fixed share), per-item weights divided by source size, epochs per source checked, choice validated by closed-loop success with seeds.</details>

4. What does decoding cost when sampling random frames with 2-second GOPs at 30 fps?
<details><summary>Answer</summary>About 30.5 decoded frames per sample on average: decode-bound loaders. Sample windows or re-encode.</details>

5. Label for intervention data?
<details><summary>Answer</summary>The executed (human) actions; the policy's proposals during takeover are what went wrong.</details>
