# Questions (out loud)

1. Why does naive off-policy Q-learning fail on a fixed dataset? Name the mechanism.
2. How does IQL avoid querying out-of-distribution actions, in each of its three losses?
3. What does the expectile τ control? What happens at τ = 0.5 and as τ → 1?
4. When does offline RL beat BC, and when is filtered BC (keep the best episodes) just as good?
5. Interviewer (Skild/PI-style): "We have 10,000 hours of teleoperation and 100 hours of autonomous rollouts with failures. How do you use both?"
6. Your offline-RL policy scores well on your Q-function and badly on the robot. Why can't you evaluate offline, and what do you do?
