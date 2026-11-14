# Questions (out loud)

1. Symmetric vs asymmetric, per-tensor vs per-channel: what does each choice cost and buy?
2. How do you calibrate activation ranges, and why not min-max?
3. PTQ vs QAT: when is post-training enough, and what does QAT cost?
4. Why does the straight-through estimator work at all?
5. Which layers of a VLA would you keep in fp16, and why?
6. Weight-only int4 (AWQ/GPTQ-style) vs int8 weights and activations: which helps a Jetson-served VLA more, and why?
