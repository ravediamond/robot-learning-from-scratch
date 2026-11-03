# Questions (out loud)

1. What is z supposed to capture in ACT, and why is it set to 0 at test time?
2. Your sweep (predict_run): describe what beta does to the information in z, with your numbers. What did you learn about running ONE seed?
3. L1 vs MSE on bimodal targets: why does L1 sometimes "pick a side"? Is that a reliable way to handle multimodality?
4. Temporal ensembling vs executing chunks open-loop (W4) vs real-time chunking: latency, smoothness, reactivity. When would you choose each on an SO-101 at 30 Hz with a 300 ms policy?
5. The review bug: how late is the arm, roughly, as a function of H? Why is the motion still smooth?
6. Interviewer: "ACT or Diffusion Policy for 50 demos of a pick-and-place?" Decide, and say what result would make you switch.
