# Lessons

What the experiments taught, written so the next decision doesn't re-learn it. Each lesson names the evidence.

## About world models on this setup

1. **A world model that predicts well can still be useless for choosing actions.** G0 passed (K=30 error 0.31 vs
   copy-last 0.89; reacts to action timing), W2 encodes the arm (R² 0.92), RoboJEPA uses our actions (9-10 % below
   shuffled). Yet G1b failed and an action-only scorer beat both world models. Prediction quality is not decision
   quality.
2. **On a position-controlled arm, the commanded joints already say where the arm will be.** The action-only baseline
   (no model, no images) reached top-1 0.68-0.90 and regret ≈ 0 on G1b. A world model only earns a role on what the
   actions can't tell: object state, contact, failures. Test it there (G7), not on arm motion.
3. **Raw latent distance is not progress.** G6a: distance to the nearest sub-goal does not fall steadily over successful
   demos (Spearman -0.32 to -0.50, bar -0.8), so stall detection by distance gives 32 % false stalls. The papers said the
   same before we tested (TD-JEPA: ~0.65 for latent distance vs 0.95 for a learned cost; ARC-Bench). A progress signal
   has to be trained on progress labels.
4. **The rollout must reach the target.** G1b on W1: rollouts shorter than the target distance rank below chance
   (H = 5 at d = 30: 0.058). Planning Limits shows the same even with a perfect simulator. Space sub-goals within the
   measured range.
5. **Spatial features beat a single pooled vector.** W1 (192-d) arm pose R² 0.57 vs W2 (DINOv2 patches) 0.92; RoboJEPA
   scaled only on V-JEPA 2.1's dense features. Never PCA-compress the latent (the information sits in low-variance
   directions).
6. **Big pretrained encoders don't fit the robot.** V-JEPA 2.1-G: 526 ms for two views on the Orin NX, 1025 ms on the
   Mac. W2's rollouts: 1.8 s per decision on the Mac. Anything in the control loop must be small; the rest runs off-board
   on CUDA.
7. **Judging is easier than imagining.** Robometer (trained on progress and success labels, >1M trajectories) passed
   zero-shot on SO-101 data; the small world models (trained only to predict the next latent, on 157 episodes) did not
   earn a selection role. The missing piece for the world model is a learned cost, not a better predictor.
   *Note 2026-10-16: on the complete set Robometer fails zero-shot (0.818), so the "passed" above held only for the
   partial set. Judging is still the closer of the two, but it is not free either.*

## About evaluation

8. **Check the confound before the model.** C1: policy identity alone (0.76) beat every latent probe on the 81 gate
   rollouts, because TurboVLA produced most failures. G9a-AB: episode length (0.934) beat Robometer (0.861; complete set 0.938 vs 0.818). In both
   cases the cheap baseline was the first thing to compute.
9. **Top-1 of N compounds errors.** Observed top-1 ≈ pairwise^7 for 8 candidates (W2 pairwise 0.92-0.93 gives top-1
   0.55-0.61). A 0.8 top-1 bar needs ~97 % per comparison. Use regret or pairwise accuracy.
10. **Recorded decoys are easier than generated ones,** and replanning can hide ranking errors (from the papers checked
    against G1b). Generated close candidates (G7b) are the harder, more honest test.
11. **Run an experiment only if one of its outcomes changes the next action.** The RoboJEPA 1B re-run was cancelled on
    that rule: a pass would still be off-robot only, a fail changes nothing.
12. **Easy contrasts prove little.** G6a T2 reached AUROC 1.00, but its negatives were at-rest frames before any grasp.
    The real question (near-misses vs successes) needs real failures.
13. **Missing data isn't random.** G9a-AB lost 22 % of episodes to one CUDA out-of-memory failure, and they were two
    whole tasks plus the longest episodes. Report errors per task. Re-scoring them flipped the verdict from qualified
    pass (0.861) to fail (0.818): a partial result is provisional until it covers every task.
14. **Ceilings hide differences.** Every working policy scored 9-10/10 on bowls, so bowls can't choose between SmolVLA and
    VLA-JEPA. Decide on the hardest primitive.

## About data and the robot

15. **Idle frames teach freezing.** TurboVLA learned "stay" from idle starts; trim every dataset the same way.
16. **Instruction-dependent behaviour needs data where the instruction matters.** TurboVLA's task-distinguishing action was
    ~6 % of frames, so it ignored the instruction ("copycat").
17. **Teleop is the scarce resource.** Two robot days a week. Everything that doesn't need the arm runs first on compute
    days; robot days must never debug software.
18. **Labels: one key press per episode, right after it.** Failures train the world model, the cost and the judge, never
    the policy.
19. **Camera choice changes the judge.** Robometer top view 0.861, front view 0.798 (both on the partial set); a badly framed "top" view (k1000dai)
    is as bad as a wrong camera.

## About cost and process

20. **Rank vast.ai offers by whole-job cost** (GPU + disk + bandwidth); interruptible bids cost more than they save on
    long runs (VLA-JEPA: $9.7 against a $3.5-4.7 estimate).
21. **Debug on the Mac, rent only for the run.** Idle debugging on a rented GPU is where money goes.
22. **Freeze the rule before the number, and label everything after as exploratory.** It is what made the G1b, G6a and
    G9a-AB results trustworthy even when they disappointed.
23. **Tokens never go in chat.** A pasted HF write token had to be revoked; credentials come from the environment only.
24. **Every checkpoint a frozen protocol depends on goes to the Hub the day it is produced.** W2 was lost with a destroyed
    vast instance (10-22); G7a was blocked and W2′ cost a retrain. *Closed 2026-10-23:* W2′ `weights_best.pt` (sha256
    `2adf3313…`) and W1 `weights_final.pt` (`cc9c5f0b…`) are in the private repo `ravediamond/lewm-w2-so101-bowls`
    ([`b5bd6036`](https://huggingface.co/ravediamond/lewm-w2-so101-bowls/commit/b5bd6036bd38e235fde4c56c3eb46948586db2d4) `w2prime/`,
    [`d0336db6`](https://huggingface.co/ravediamond/lewm-w2-so101-bowls/commit/d0336db6e13d7477d72c33261e0892969fc035c0) `w1/`).
