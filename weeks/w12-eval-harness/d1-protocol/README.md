# W12 D1: the protocol: schedule, blinding, records

> **Full lesson for this day:** [`../README.md`](../README.md#day-1-the-protocol) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/13-evaluation-at-scale.md` §1.

## The one idea

A rollout's outcome depends on the policy AND on everything else that changes during a session:
the battery, the light through the window, the cube that wears out, the operator who gets better at
resets. If you run 50 rollouts of policy A in the morning and 50 of B in the afternoon, you measured
the morning against the afternoon.

Three rules remove these confounds:

1. **Fixed start configurations.** Mark 10 object positions (tape on the table). Every policy runs from
   every start the same number of times. Then comparisons are **paired** (W6 D4).
2. **Blocked, counterbalanced schedule.** A **block** = one start configuration, every policy once.
   Inside a block the order rotates (a balanced Latin-square rotation), so no policy is always first.
   Drift then hits every policy equally.
3. **Blinding.** The operator sees aliases ("A", "B", "C"), never names. They decide success, reset
   the scene and sometimes intervene: all of that bends towards the policy they hope wins.

And a record per rollout with a **fixed schema**, validated when written, not when analysed.

## Today's build (`w12d1_protocol.py`)

`make_schedule`, `blind_aliases`, `validate_record`, `expected_success`.

## Staff angle

"How did you control for drift and operator bias?" is the first question a research lead asks about a
real-robot table. The answer "blocked, counterbalanced, blinded, fixed starts, schema-validated logs"
in one breath is what makes your numbers believable before anyone reads them.
