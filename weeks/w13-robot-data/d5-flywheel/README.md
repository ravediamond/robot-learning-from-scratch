# W13 D5: the data flywheel + design

> **Full lesson for this day:** [`../README.md`](../README.md#day-5-the-data-flywheel-a-design) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/14-robot-data-engineering.md` §5.

## The one idea

Once a policy runs, its own rollouts are the most valuable data you can get: they show the states the
policy actually reaches. The flywheel turns them into training data:

1. **Interventions.** When an operator takes over, the human's corrective actions from that moment are a
   demonstration of recovery (DAgger, W6 D5; HIL-SERL). Extract each takeover segment with a few
   frames of context before it, and label it with the **executed** (human) actions.
2. **Failure mining.** Rank autonomous episodes for human review by a failure-detector score and by
   uncertainty; review the top ones first.
3. **What to collect next.** Allocate the next N demonstrations across start conditions in proportion
   to where the policy fails and where you are least sure of the success rate.

Then **design #2**: the data platform for a fleet.

## Today's build (`w13d5_flywheel.py`)

`intervention_segments`, `correction_dataset`, `review_queue`, `allocate_collection`.

## Staff angle

This is the loop that Physical Intelligence (RECAP), HIL-SERL and every deployed-robot company runs.
The engineer who can say how many hours per week of review it costs and what it bought in success rate
is the one who owns it.
