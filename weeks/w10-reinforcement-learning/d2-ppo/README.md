# W10 D2: PPO

> **Full lesson for this day:** [`../README.md`](../README.md#day-2-ppo) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/11-reinforcement-learning.md` §3. Paper: Schulman et al., 2017.

## The idea
REINFORCE uses each batch once. PPO reuses it for several epochs, and stops the policy from moving too
far from the one that collected the data. With the probability ratio `ρ = π_new(a|s) / π_old(a|s)`
(computed as `exp(logp_new − logp_old)`):
```
L_clip = −mean( min( ρ · A ,  clamp(ρ, 1−ε, 1+ε) · A ) )
```
The `min` makes the objective **pessimistic**: once ρ has moved past the clip in the direction that
helps (ρ > 1+ε with A > 0, or ρ < 1−ε with A < 0), the gradient is zero. Moving the wrong way is never clipped.

The rest is bookkeeping that decides whether it works:
- advantages from GAE, **normalised per minibatch** (zero mean, unit std), always detached;
- value loss `0.5 · (V − target)²` with `target = advantage + old value`;
- `logp_old` and the old values are stored at collection time, **without gradient**;
- several epochs over shuffled minibatches; small entropy bonus optional.

## Today's build (`w10d2_ppo.py`)
`ppo_clip_loss`, `normalize`, `value_loss`, `ppo_update`, `train_ppo` on `Reach2D`.
Yesterday's `discounted_returns` / `gae` are provided in `w10d2_helpers.py` so today stands alone.

Run: `uv run pytest weeks/w10-reinforcement-learning/d2-ppo`
