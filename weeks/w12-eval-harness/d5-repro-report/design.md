# Design: the evaluation program for a generalist manipulation policy

**Timer: 45 minutes. Write in `my_design.md`. Then grade with `rubric.md`.**

## Prompt

Your startup trains one VLA for home and light-industrial manipulation. Every two weeks, research
produces 3 to 5 candidate checkpoints. You have 12 identical robot stations across 2 sites, 5 target
tasks (from pick-and-place to folding), operators on day shifts, a simulator that covers 2 of the 5 tasks
badly, and a world model that researchers want to use as an offline evaluator.

Design the evaluation program: what is measured, how, by whom, at what cost, and how a checkpoint gets
promoted to the fleet.

## What the interviewer will push on

- Robot hours per release, and how you choose which checkpoints get them.
- Statistical power: what difference can you detect per task, and what do you do about the tasks you cannot power?
- Drift across stations and sites: how do you know a station is broken?
- Who labels success and failure causes, and how do you know the labels are consistent?
- Sim and world-model evaluation: when do you trust them, and how do you prove it?
- Latency and deployment settings: are you evaluating what will run in the field?
- The research team wants results in 24 hours. Operations wants zero regressions in the field.
- What do you report, and to whom?

## Deliverable

Goals and non-goals → evaluation tiers (offline, sim/world model, real) with what each decides →
protocol → statistics and budget → infrastructure (scheduling, logging, manifests, dashboards) →
promotion rule → failure analysis loop → what you would not build in v1.
