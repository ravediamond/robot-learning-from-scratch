# Questions (out loud)

1. Why does the residual connection make deep transformers trainable? Say it in terms of gradients.
2. Learned absolute vs sinusoidal vs RoPE: which one would you pick for an action expert whose chunk length is fixed at 50? For a VLM that sees variable numbers of image tokens?
3. The review bug: what happens to the signal after 24 such blocks? Why is the loss flat rather than NaN?
4. A ViT on 224×224 with p=16 for 2 cameras and 2 frames: token count? What would you do to fit an Orin's budget?
5. Why GELU and not ReLU in transformers? Is it important?
6. Interviewer: "Draw the full SmolVLA forward pass, with token counts per part." (Use primer 03.)
