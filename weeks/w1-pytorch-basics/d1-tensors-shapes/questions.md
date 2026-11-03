# Questions (answer out loud, 1–2 min each)

1. `view` vs `reshape`: what is the difference, and when does `view` fail? What is "contiguous"?
2. What is a stride? Why is `unfold` free (no copy) and why does that matter for a 1M-frame dataset?
3. float32, float16, bfloat16: what does each trade? Why do most robot policies train in bf16 but keep the optimizer state in fp32?
4. The review bug did not crash on your 6-joint arm with batch 6. On which shapes would it crash? Why is a non-crashing shape bug the worst kind?
5. You get an episode dataset with lengths 80 to 900 steps. Pad everything to 900, or bucket by length, or sample fixed windows? What does each cost?
6. Interviewer: "Your state has 6 dims and the gripper std is 0 on the training set. What happens at inference without eps? With eps?" (Then: what is the better fix?)
7. Why does a chunked policy (ACT, SmolVLA) predict `(H, D)` instead of `(D,)`? Give two reasons. (Read book chapter 11 if stuck.)
