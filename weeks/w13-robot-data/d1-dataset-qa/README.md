# W13 D1: dataset QA at scale

> **Full lesson for this day:** [`../README.md`](../README.md#day-1-dataset-qa-at-scale) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/14-robot-data-engineering.md` §1. W2 D4 checked one episode by hand (gaps, idle, lag).

## The one idea

A 2,000-episode dataset always contains episodes that teach the wrong thing: a camera that froze for
eight seconds, a joint whose encoder died, an arm driven into its limit, a calibration that changed
halfway, the same episode uploaded twice, a 3-second "episode" that was a mis-click. Nobody watches
2,000 videos. You write **checks**, run them on every upload, and **quarantine** with a reason.

Checks that pay for themselves:

| Check | Signal |
|---|---|
| frozen camera | the longest run of identical consecutive frame hashes |
| dead joint | a joint whose range over the episode is ~0 while others move |
| limit hits | fraction of frames at (or beyond) the calibrated joint limits |
| length outliers | robust z-score with the **median and MAD**, not mean and σ |
| duplicates | identical content hash of (state, action) arrays |

Mean ± 3σ fails at exactly this job: one 40-minute episode inflates σ so much that it hides itself and
everything else ("masking"). The median absolute deviation (MAD) does not move.

## Today's build (`w13d1_qa.py`)

`longest_frozen_run`, `dead_joints`, `limit_fraction`, `robust_z`, `content_hash`, `qa_report`.
`fake_fleet.py` (provided) generates episodes with known defects.

## Staff angle

The research-engineer move is not "I cleaned the data". It is "every upload runs 7 checks, 3.1% of
episodes were quarantined last month, half of them from one station whose wrist camera USB drops,
and here is the policy result with and without them".
