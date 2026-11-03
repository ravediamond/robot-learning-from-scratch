# Questions (out loud)

1. State the broadcasting rule in one sentence. Then give a case where broadcasting silently produces the wrong shape instead of an error.
2. The review bug: the policy still trains. What exactly goes wrong? Which episodes are hurt most, and how would it show up on the robot? (Hint: the end of an episode is where the grasp/place happens.)
3. `pairwise_sq_dist` on N = M = 100k: what is the memory of the broadcast version? How would you do it in chunks? What is the trick `||a||² + ||b||² − 2a·b` and its numerical risk?
4. Best-of-N: you sample N chunks from a policy and pick the one a scorer likes best. Name two reasons this can fail even with a good scorer. (Then compare with your H0 result in `reference/journal/EXPERIMENTS.md`.)
5. `gather` vs `x[torch.arange(B), idx]`: when do you need `gather`?
6. Why are in-place ops dangerous with autograd? (You will see the error tomorrow.)
