# 08. Question bank: mixed mock interview

Use in week 8 (and any Friday). Pick 5 at random, 2 minutes each, record yourself. No
answers here on purpose: answer from the primers and the book, then compare. Book chapter
16 and `reference/journal/book/exercises/interview/` hold the rehearsal scripts.

## A. Your work (they WILL go here first)
1. Walk me through the system you built. What is the one result you are most sure of, and why?
2. What did you try that did not work? (Best-of-N reranking H0; TurboVLA port; scale as verifier.) What did each teach you?
3. Which of your published numbers would you not bet on today? Why?
4. Tell me about the checkpoint-loading bug. How was it found, what did it invalidate, what did you change in the process?
5. You use Claude Code to run experiments. How do you know its results are right? Give one case where it was wrong.
6. Why SO-101 and an Orin? What can and can't you conclude from cheap hardware?
7. Pick one of your paper drafts. Main claim, evidence, strongest objection.

## B. Research reasoning
8. A paper reports +15% success with a new world-model-based reranker. Design the experiment that would make you believe it on your arm. What is the cheapest baseline?
9. Your latent world model's prediction loss keeps going down but planning success doesn't change. Hypotheses, in order, and the test for each.
10. JEPA vs video generative world model for a startup with 3 robots and 1 GPU node. Choose and defend.
11. When is RL fine-tuning on the real robot worth it compared with collecting more demos?
12. How would you build a reward model for a task without a ground-truth success signal?
13. What would make you abandon a research direction after one week?

## C. System design
14. Design a data flywheel: robots in a customer site, failures detected, data curated, policy retrained, safely redeployed. Where are the humans? What is measured at each stage?
15. Inference on edge: VLA at 300 ms on an Orin, need 30 Hz control. Architecture?
16. Training infrastructure for 50k episodes of multi-camera data: storage format, loading, sharding, eval.
17. Scale training 1 → 8 GPUs. What changes (batch, lr, sampler, BN, logging, checkpoints)?
18. Design the evaluation suite for a household long-horizon task that runs every night.

## D. Fundamentals, quick fire (30 s each)
19. Why √d in attention? 20. What is the KV cache? 21. Why chunk actions?
22. MSE vs flow matching on multimodal data. 23. Collapse in JEPA, three fixes.
24. EMA target encoder: what and why? 25. Wilson vs normal interval.
26. Why split by episode? 27. Compounding error and DAgger. 28. RoPE in one sentence.
29. LayerNorm vs BatchNorm for policies. 30. Temporal ensembling vs real-time chunking.
31. CEM vs gradient-based planning. 32. What does a linear probe tell you, and not tell you?
33. FK vs IK, why is IK harder? 34. Hand-eye calibration in one sentence.

## E. Pushback (the interviewer disagrees; hold your ground or update, with reasons)
35. "World models are a distraction; just scale demos."
36. "Your hardware is a toy; nothing you found transfers."
37. "You don't have a PhD or papers; why should we trust your research judgment?"
38. "Statistics are overkill; we just watch the robot."
39. "Your eval method is slow; startups need speed."
