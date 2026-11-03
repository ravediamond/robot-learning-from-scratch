# Questions (out loud)

1. You measured the action → state lag on your data. What does it mean for (a) the hold-position baseline, (b) chunk size, (c) a policy that sees only the current state?
2. You find 3 episodes with timestamp gaps of 10 frames. Drop them, keep them, or interpolate? What does each do to training and to evaluation?
3. Why does training on idle starts teach a policy to freeze? Why is "just trim them" not entirely free either?
4. The review bug: why did it work on one machine and not another? What is the general lesson about relying on row order?
5. Your dataset is 61k frames, 2 cameras at 30 fps. Estimate the disk size of raw RGB frames at 480×640 vs the mp4s. Why does that matter for the data loader?
6. Interviewer: "How do you know your teleop data is good before training on it?" Give five concrete checks, with numbers you would look at.
