# Questions (out loud)

1. Why can you jump to any x_t in closed form instead of noising step by step? Derive abar.
2. Why predict the noise rather than x_0 directly? (Hint: the target's scale at different t.)
3. The review bug: why does the loss still go down? What does the model learn to do?
4. DDIM vs DDPM sampling: what changes, and why does Diffusion Policy use ~10–16 DDIM steps at inference?
5. What does the denoiser look like for an action chunk (Diffusion Policy): inputs, conditioning, architecture?
6. Interviewer: "Diffusion policy is too slow for our 50 Hz arm. Options?"
