# Questions (out loud)

1. Linear probe low, MLP probe high: what does it mean, and is it a problem for (a) planning, (b) a linear reward head, (c) a VLA reading the latent?
2. Why must probes be evaluated on held-out EPISODES? (You know the frame-split trap.)
3. What is the right "random encoder" baseline for a probe, and why is it surprisingly strong for images?
4. The review bug: why did every probe look excellent?
5. Copy-last vs predictor error at horizon 1 and at horizon 10: which comparison is more informative, and why?
6. Interviewer: "Your JEPA latent has probe R² 0.95 for gripper position. Is it a good world model?" (What is missing from that claim?)
