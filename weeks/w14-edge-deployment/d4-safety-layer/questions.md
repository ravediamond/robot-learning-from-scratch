# Questions (out loud)

1. List the layers of safety between a learned policy and a person standing next to the robot.
2. What belongs in a deterministic safety filter, and what does not?
3. Why must an e-stop latch? What resets it, and who?
4. Timestamps: capture, receive, inference start, command sent. Which ones do you log and what do you compute from them?
5. The filter holds the last command when the observation is stale. When is "hold" the wrong behaviour?
6. How do you test a safety layer, given that it must never fail?
