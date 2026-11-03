# Questions (out loud)

1. Derive why q·k has variance d when entries are i.i.d. N(0, 1). What does it do to softmax gradients?
2. Mask with −inf vs with a large negative number (−1e9): what are the trade-offs, especially in fp16?
3. The review bug: which token breaks first and why? What does the model lose even before the NaN appears?
4. Attention is permutation-equivariant (predict 6). So how does a transformer know token order? What about a VLA's image patches?
5. Memory of attention for T = 4,096, 16 heads, fp16, batch 8: compute it. What does FlashAttention change?
6. Interviewer: "Write attention with a padding mask, then tell me the bug you would most likely make." (Say it: wrong mask polarity, wrong broadcast dim.)
