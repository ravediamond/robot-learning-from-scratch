# Week 10: Reinforcement learning, from returns to RL on top of imitation

Primer: `knowledge/11-reinforcement-learning.md`.

Almost every robot-learning posting names RL next to imitation learning, and the 2025–2026 frontier
result (π*0.6 with RECAP) is "imitation first, then improve with RL from the robot's own experience".
This week you write the pieces by hand, small enough to run on a Mac in seconds.

All labs share one toy world, `rl_envs.py` (provided): `Reach2D`, a batch of 2-D points that must reach
a goal. obs = (pos, goal), action = velocity in [-1, 1]², reward = −distance, fixed horizon of 20 steps.

| Day | Lab | Review bug |
|---|---|---|
| d1 | returns, GAE, REINFORCE with a baseline | returns leak across the episode boundary |
| d2 | PPO: clipped objective, advantage normalisation, the update loop | `max` instead of `min` in the clip |
| d3 | SAC: tanh-Gaussian log-prob, twin Q target, soft update, temperature | missing tanh correction |
| d4 | offline RL: IQL (expectile value, advantage-weighted policy) beats BC on mixed data | expectile weights swapped |
| d5 | RECAP-style advantage conditioning on top of imitation | one global threshold for all tasks |
