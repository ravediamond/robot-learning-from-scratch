# Design: the data platform for a robot fleet

**Timer: 45 minutes. Write in `my_design.md`. Then grade with `rubric.md`.**

## Prompt

100 SO-101-class arms at 3 sites collect teleoperation data 8 hours a day (3 cameras each) and, from next
quarter, run policies autonomously with operators who can take over. Researchers train on the data
weekly and want any past dataset version reproducible. Customers' sites have flaky networks and some of
their scenes must never leave the EU.

Design the data platform: ingestion, storage, quality, curation, versioning, access for training, and the
loop from deployment back to training.

## What the interviewer will push on

- Daily volume, storage per year, and what you keep at which quality.
- What happens between a robot finishing an episode and the episode being usable for training?
- How do you catch a broken camera at site 2 within an hour, not a month?
- How does a researcher get "the dataset as of training run 412" in a year?
- Interventions and failures: how do they flow back, and who labels them?
- Training-side throughput: how many GPUs can your loaders feed?
- Privacy and residency: faces on camera, EU-only scenes, deletion requests.
- What would you NOT build in the first three months?

## Deliverable

Requirements and numbers → architecture (edge, ingestion, storage tiers, catalogue, QA, curation, training
access, flywheel) → one episode's path → versioning → monitoring → privacy → v1 scope. Grade with `rubric.md`.
