# Questions (out loud)

1. Write the flow-matching training step and the Euler sampler from memory, with shapes. (Coding list 11–12: 10 + 8 min.)
2. Explain the 1-step collapse to the mean in terms of what the network can know at t = 0.
3. Why is the loss not near 0 at convergence? What is the network actually regressing?
4. Straight paths (rectified flow) vs diffusion paths: why do straight paths need fewer sampling steps?
5. The review bug: describe what the trained network learns and what the sampler then does with it.
6. Interviewer: "How would you make your flow policy 1-step at inference without losing multimodality?"
