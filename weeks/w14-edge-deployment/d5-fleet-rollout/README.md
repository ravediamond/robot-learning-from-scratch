# W14 D5: rolling a new policy out to a fleet + design

> **Full lesson for this day:** [`../README.md`](../README.md#day-5-fleet-rollout-a-design) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/15-edge-deployment.md` §5.

## The one idea

A new policy that won the evaluation can still lose in the field (other lighting, other objects, a site
nobody evaluated). Roll it out like a risky software release:

1. **Artifact check**: the robot loads exactly the approved weights (fingerprint match, W12 D5).
2. **Shadow mode**: the new policy runs on every observation but its actions are not executed; compare
   them with the deployed policy's actions. Large disagreement rates point to situations to look at.
3. **Canary**: a few robots run the new policy; compare their **intervention rate per hour** with the rest.
   Compare rates, not counts (5 canary robots vs 45 others), and test whether a higher rate is more than
   noise before rolling back (conditional binomial test on Poisson counts).
4. **Staged rollout** with an automatic rollback rule decided in advance.

## Today's build (`w14d5_rollout.py`)

`verify_artifact`, `shadow_disagreement`, `rate_ratio_pvalue`, `canary_decision`, `stage_sizes`.

## Staff angle

"The canary had 3 interventions in 40 hours against 30 in 400 hours: the same rate, p = 0.6, continue"
is the sentence that keeps a team from rolling back good models on noise, or keeping bad ones on hope.
