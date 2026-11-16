# W10 D4: Offline RL with IQL

> **Full lesson for this day:** [`../README.md`](../README.md#day-4-offline-rl-with-iql) (the week page: concepts from zero, examples, traps).
> This file is the one-page summary. Order of work: `predict.md` → the lesson → `predict_run.py` → the exercise → the review file → `questions.md`.

Primer: `knowledge/11-reinforcement-learning.md` §5. Paper: Kostrikov et al., 2021 (IQL).

## The problem
You have a fixed dataset from mixed sources: good teleoperation, sloppy teleoperation, failed
autonomous rollouts. **BC imitates the average**. Naive Q-learning on fixed data fails differently: the
target queries `Q(s', a')` for actions the dataset never contains, and those values are fantasy.

## IQL: never ask about unseen actions
Three networks, all trained only on dataset `(s, a, r, s')`:
1. **V(s) by expectile regression** towards Q(s, a) of dataset actions. With τ > 0.5, V leans to the
   value of the BEST actions seen in that state, without ever taking a max over unseen ones.
   ```
   L_V = mean( |τ − 1[d < 0]| · d² ),   d = Q_target(s, a) − V(s)
   ```
2. **Q(s, a)** by a normal TD loss using V for the next state: `y = r + γ (1 − done) V(s')`.
3. **Policy** by advantage-weighted regression: imitate dataset actions, weighted by how good they were:
   ```
   w = min( exp(β · (Q(s, a) − V(s))), w_max )         L_π = mean( w · ||π(s) − a||² )
   ```

## The dataset (provided, `offline_data.py`)
Reach2D episodes: 25% expert, 25% random, 50% an "anti-expert" that walks away from the goal.
The average action points AWAY from the goal: BC is worse than doing nothing.

## Today's build (`w10d4_iql.py`)
`expectile_loss`, `awr_weights`, `train_bc`, `train_iql`.

Run: `uv run pytest weeks/w10-reinforcement-learning/d4-offline-rl-iql`
