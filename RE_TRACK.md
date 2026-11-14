# Track: research engineer, data / evaluation / deployment angle

**Pitch: an engineer who makes robot learning reproducible, measurable and deployable.**

Research-engineer roles in robot learning split in practice into flavours. The model-research flavour
(world models, new architectures) screens paper depth and training at scale, where PhDs compete. The
data / evaluation / real-robot flavour screens the ability to build the machinery that tells a research
team what is true and to make policies run on real arms. This repo covers both; this page is the order
of work for the second, where years of data, ML and evaluation engineering count.

Every loop still has a live PyTorch or ML round: the basics are not optional.

## What the loops test, and where it is trained

| Round | What it checks | Labs |
|---|---|---|
| Live PyTorch coding | attention, losses, sampling, training step, shapes out loud | `drills/live` L01–L10, W1, W3, W4 |
| Fix a training script | finding real bugs fast | W9 D3, every `review_*.py` |
| ML / robot-learning depth | IL vs RL, flow vs MSE, VLA anatomy, offline data | W4–W7, W10, primers 01–07, 11 |
| Evaluation and statistics | paired protocols, CIs, power, failure analysis | W6, W12, primers 07, 13 |
| Data | QA, curation, splits, mixtures, storage, flywheel | W2 D4, W13, primer 14 |
| Deployment | latency on target, export, quantisation, RTC, safety | W8 D3, W12 D4, W14, primer 15 |
| Design | data platform, evaluation program, fleet deployment | W9 D5, designs in W12 D5, W13 D5, W14 D5 |
| Project talk | your own work, defended | `flagship/` + your project page |

## Fast track (about three weeks of evenings and weekends)

| Order | Do | Why first |
|---|---|---|
| 1 | `drills/live` L01, L03, L06, L07 (timed, AI off), redo the misses 48 h later | the live round is the first filter |
| 2 | W12 d1–d5 + the flagship dry run (`flagship/README.md` step 3) | your flagship's engine, and the eval round |
| 3 | `flagship/` pilot: 2 policies × 20 rollouts, protocol v2 | a real result beats a perfect plan |
| 4 | W13 d1, d2, d5 | the data round and the data-flavoured roles |
| 5 | W14 d1, d3, d4 | deployment questions; RTC and safety are high-signal |
| 6 | `drills/live` L02, L04, L05, L08–L10; W10 d1–d2 | breadth for the live round; RL screen basics |
| 7 | Designs: W12 D5, W13 D5, W14 D5 (45 min each, graded) | the design round |
| 8 | `flagship/` full run, then the page | the portfolio piece |

Apply when step 3 is done, with the pilot on your project page and the full run announced as in
progress. Do not wait for step 8.

## The flagship in one paragraph (say it in 30 seconds)
Four policies (ACT, SmolVLA, VLA-JEPA, my own head) on the same SO-101 task and data, 50 rollouts each
on 10 marked starts, blocked and counterbalanced, operator blind; success with Wilson intervals,
pairwise McNemar with Holm, the minimum detectable difference stated up front, failure causes with
rater agreement, a stage funnel, and latency percentiles with the async-chunking budget on the Jetson
Orin. Every number has a manifest and the analysis code is public.
