# Questions (out loud)

1. Write the PPO clipped objective from memory and explain the `min` with one example for A > 0 and one for A < 0.
2. Why are advantages normalised per minibatch, and when can that hurt?
3. What must be stored at collection time, and what breaks if you recompute it during the update?
4. PPO needs millions of steps for locomotion. Why is that acceptable in Isaac Lab and not on your SO-101?
5. Your PPO return rises, then collapses. List four things to check, in order.
6. PPO vs SAC: which would you pick for (a) a legged robot in a GPU simulator, (b) fine-tuning a manipulation policy on a real arm? Why?
