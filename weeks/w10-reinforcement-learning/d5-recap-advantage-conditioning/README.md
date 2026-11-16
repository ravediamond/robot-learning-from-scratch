# W10 D5: RECAP-style advantage conditioning

> **Full lesson for this day:** [`../README.md`](../README.md#day-5-recap-style-advantage-conditioning) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/11-reinforcement-learning.md` §6. Paper: Physical Intelligence, "π*0.6: a VLA That Learns
From Experience" (arXiv 2511.14759).

## The idea
You cannot easily run PPO or SAC on a 3-billion-parameter flow-matching VLA. RECAP sidesteps policy
gradients entirely. It keeps the supervised recipe and changes the **conditioning**:

1. Train a **value function** on all the data (demos, autonomous rollouts, interventions).
2. For every step compute an **advantage**: was this action better than usual from this state?
   `A_t = (Σ_{k<n} γ^k r_{t+k}) + γ^n V(s_{t+n}) − V(s_t)`  (n-step; cut at the episode end).
3. Turn it into one bit, `I_t = A_t > threshold`, where the threshold is a quantile **per task**.
4. Train the policy on ALL data with `I_t` as an extra input, sometimes dropped (so the policy also
   learns the unconditional behaviour, as in classifier-free guidance).
5. At test time, ask for `I = 1`: "act like the better-than-usual part of the data".

No data is thrown away, bad data is still used (the model learns what `I = 0` looks like), and the
training loss stays plain imitation.

## Today's build (`w10d5_recap.py`)
`fit_value`, `n_step_advantage`, `improvement_indicator`, `train_conditioned`, on yesterday's sources mixed
**per step** (`make_dataset(per_step=True)`): at every step the actor is the expert (25%), random (25%) or the
anti-expert (50%). The state no longer reveals who is acting; only the advantage can tell good steps from bad.
The policy here is an MSE head; in π*0.6 it is the flow head, with the same conditioning idea.

Run: `uv run pytest weeks/w10-reinforcement-learning/d5-recap-advantage-conditioning`
