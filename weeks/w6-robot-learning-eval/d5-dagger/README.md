# W6 D5: Compounding error and DAgger (capstone, part 2)

Book: 11.1 (BC from first principles). Primer 03 (compounding error). Uses `dagger_world.py` (read it).

## The problem
BC learns from the expert's state distribution. A small error or a disturbance puts the robot in a
state the expert never visited; there the policy is untrained and its next error is bigger; errors
compound over the episode (Ross & Bagnell: the cost of BC can grow with T² in the horizon).

## DAgger (Dataset Aggregation)
```
D = expert demos;  π = train(D)
repeat:
    run π (WITH the disturbances it will face) and record the states it visits
    label those states with the EXPERT's action           <- the key line
    D = D ∪ {(state, expert(state))};  π = train(D)
```
On a real robot "the expert labels the states the policy visits" = human corrections / interventions
(HG-DAgger), the recovery data you collect by pushing the arm off course during teleop.

## Today's build (`w6d5_dagger.py`)
`train_bc`, `run` (rollouts with the push at `KICK_T`, optionally recording visited states),
`aggregate`, `dagger`.
Test: plain BC on 10 demos < 70% success under pushes; DAgger after 2 iterations > 90%, paired on
the same 200 starts.

Run: `uv run pytest weeks/w6-robot-learning-eval/d5-dagger -s`
