# Questions (out loud). Friday: connect it to your robot.

1. Give your three real-data numbers (hold, state-only, pixel). Is the gap as big as on the toy? Explain the difference with what you know about the bowls task.
2. Blank-image test (predict item 3): what does it measure, and why is it a better check that "the image is used" than comparing val losses?
3. The review bug: why does val look better than train? Name the general rule it breaks.
4. You have two cameras (top, wrist). Design the ablation that tells you which one the policy relies on. How many episodes and what bar?
5. Decoding speed from extract_frames.py vs your training step time: what does this say about how LeRobot/your harness should load data?
6. Interviewer: "Your pixel policy val MSE is great but the robot misses the bowl by 2 cm." Five hypotheses, ranked by how fast you can check them.
