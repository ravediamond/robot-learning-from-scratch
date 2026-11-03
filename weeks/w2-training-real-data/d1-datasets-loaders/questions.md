# Questions (out loud)

1. Why one sample per frame and not one per episode? What does each choice do to how often long vs short episodes are seen?
2. Pad the end of an episode by repeating the last action, by zeros, or drop those windows? What does each teach the policy about "stopping"?
3. The review bug: which frames are corrupted, how many per episode, and what does the policy learn from them? Why does the loss "look normal"?
4. `num_workers > 0` and random augmentation: what goes wrong with seeding, and how do you fix it (`worker_init_fn`)?
5. Your data loader takes 40 ms per batch, the model step 10 ms. What do you do, in order?
6. Interviewer: "You have 2 tasks with 90 and 10 episodes. Balance or not?" Argue both sides, then decide.
