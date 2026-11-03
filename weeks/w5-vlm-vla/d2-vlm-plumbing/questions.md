# Questions (out loud)

1. Pixel shuffle vs average pooling of tokens vs a learned resampler (Perceiver / Q-Former): what does each keep and cost?
2. Why can the prefix be cached in a π0/SmolVLA-style mask? What would break the caching?
3. The review bug: which patches end up in one token? Why is it worse than just a lower resolution?
4. Why does a VLA put the state token AFTER the images and text, in its own block?
5. You add a third camera. What changes in the sequence, the mask, the latency, and the training data requirements?
6. Interviewer: "Explain the attention mask of π0 and why actions attend bidirectionally among themselves."
