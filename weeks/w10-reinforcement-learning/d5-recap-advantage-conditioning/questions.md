# Questions (out loud)

1. Describe RECAP in five steps. Which part is "RL" if the policy loss is still imitation?
2. Why condition on a binarised advantage instead of weighting the loss by exp(β·A) as in AWR?
3. Why is the threshold a per-task quantile? What else must the value function be given to make advantages comparable?
4. What data does this method use that pure imitation would discard, and what does each kind contribute?
5. Where can it fail? Give three failure modes and how you would detect each.
6. You own an SO-101 and 50 autonomous rollouts with a 60% success rate. Design the smallest RECAP-style experiment worth running, with its baseline and its evaluation.
