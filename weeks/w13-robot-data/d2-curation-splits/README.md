# W13 D2: curation, splits, and versioning

Primer: `knowledge/14-robot-data-engineering.md` §2.

## The one idea

Three ways a dataset lies to you after QA:

1. **Near-duplicates.** The operator records the same motion 30 times from the same start. Resample
   each trajectory to a fixed length and compare: near-identical trajectories inflate the dataset and,
   split randomly, put copies of one demonstration on both sides of the train/validation split.
2. **Leaky splits.** Episodes from the same session share lighting, objects and operator habits.
   A random episode split measures memorisation of the session. Split by **group** (session, station,
   operator, or day): a group lives on one side only.
3. **Untraceable versions.** "The dataset" changes every week. A dataset **fingerprint** (a hash of the
   sorted per-episode content hashes) and a **removal ledger** (every episode dropped, by which rule,
   when) make any training run reproducible and every curation decision reversible.

## Today's build (`w13d2_curation.py`)

`resample`, `near_duplicates`, `group_split`, `dataset_fingerprint`, `Ledger`.

## Staff angle

"We hold out sessions, not episodes, and our validation numbers dropped 15 points, which is the real
generalisation gap" is the kind of finding that changes a research team's roadmap. Be the person who
finds it, not the one who reports the inflated number.
