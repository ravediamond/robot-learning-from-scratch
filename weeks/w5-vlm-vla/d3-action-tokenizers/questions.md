# Questions (out loud)

1. Why do autoregressive VLAs with naive binning struggle at high control frequencies? Two reasons.
2. Why does the DCT compress smooth chunks, and what kind of action signal resists it (predict item 3)?
3. Per-dimension bins with fixed [low, high] vs quantile bins from the data: trade-offs?
4. The review bug: size of the bias with 256 bins over ±150°? Would you see it in val loss?
5. Tokens (π0-FAST) vs flow head (π0): if you had to pick for a new 6-DoF arm at 30 Hz with 200 demos, which and why?
6. Interviewer: "How would you tokenize the gripper differently from the joints?"
7. The test needs 10 coefficients, not 4, for signals with less than one cycle. Where in the chunk is the error largest, and why? (Hint: what the DCT assumes beyond the chunk's ends.)
