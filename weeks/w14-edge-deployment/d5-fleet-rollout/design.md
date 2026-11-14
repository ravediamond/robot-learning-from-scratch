# Design: deploy a VLA to 20 robots in a warehouse

**Timer: 45 minutes. Write in `my_design.md`. Then grade with `rubric.md`.**

## Prompt

A logistics customer wants 20 single-arm stations picking items from totes into boxes, 2 shifts a day. Your VLA
(500M parameters, flow-matching action head, 30 Hz, chunks of 50) reaches 85% success in your lab. Each station
has a Jetson Orin, two cameras and a local operator who supervises 5 stations. The network to your cloud is
unreliable. You ship a new model every month.

Design the deployment: runtime on the robot, latency budget, safety, monitoring, updates and rollbacks, and the
loop back to training.

## What the interviewer will push on

- Latency budget on the Orin: what runs where, at what precision, sync or async?
- What happens when inference is slow, a camera freezes, or the policy outputs garbage?
- How do you know, from your office, that station 14 is doing worse since Tuesday?
- How does a new model reach the 20 stations, and how fast can you undo it?
- How does what happens on site become training data?
- 85% in the lab: what do you expect on site and how do you find out early?

## Deliverable

Requirements and numbers → on-robot runtime (diagram) → latency budget → safety layers → fleet monitoring →
release process → data loop → v1 scope. Grade with `rubric.md`.
