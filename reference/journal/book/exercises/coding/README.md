# Live-coding list: what to be able to write cold

**Target role: Research Engineer, World Models, at a robotics startup** (typical JD: implement and reproduce SOTA world-model baselines; own training, data and evaluation pipelines; latent-space planning and MPC; profile, debug and scale distributed training; JEPA named as highly valued). The order below is weighted for that posting. The Data role (Research Scientist/Engineer, Data) is a different interview; see the last section.

**Short on time? Book chapter F (`../../00a-the-fast-track.md`) gives the daily routine and the priority cut.** **New to deep learning? Start with book chapter P1 (`../../00p1-deep-learning-from-zero.md`).** Concepts and APIs for every item: book chapter P2 (`../../00p2-pytorch-for-the-coding-round.md`), with a map from item number to sections in P2.17. Read the sections for an item before its first attempt, not during.**

Rules for every item: empty file, PyTorch or NumPy, no autocomplete, no AI, **no documentation** (running out of time looking up syntax is a reported failure), timer on. Write it, run it on a toy tensor, assert the shape, check one value by hand. Say the shapes out loud while typing (that is what is scored). Keep each solution in this folder as `NN_name.py` with the time it took in the first line. Redo any item that took longer than its budget until it fits.

## Tier 1 — asked at nearly every RE loop (do these first, all under 40 lines)

| # | Component | Budget | Done when |
|---|---|---|---|
| 01 | Scaled dot-product attention with causal mask | 10 min | output shape (B,T,D); rows of softmax sum to 1; future positions get zero weight |
| 02 | Multi-head self-attention module (split heads, project, merge, output proj) | 20 min | matches `torch.nn.MultiheadAttention` on the same weights within 1e-5 |
| 03 | Training loop: dataloader → forward → loss → `zero_grad` → `backward` → `step`, with eval mode and a held-out loss | 15 min | loss on one repeated batch goes to ~0 (the overfit-one-batch test) |
| 04 | Cross-entropy from logits by hand (log-softmax, gather, mean) | 8 min | equals `F.cross_entropy` |
| 05 | MSE / Huber / L1 losses, with masking of padded steps | 8 min | padded steps contribute exactly zero |
| 06 | SGD with momentum and Adam as classes (`step`, `zero_grad`) | 15 min | one step equals `torch.optim.Adam` on the same grads |
| 07 | Layer norm, RMS norm, forward only | 8 min | matches `nn.LayerNorm` |
| 08 | Linear layer forward + manual backward (dW, db, dx) | 15 min | equals autograd |
| 09 | A tiny MLP classifier trained end to end on random separable data | 15 min | accuracy > 95 % |
| 10 | Cosine learning-rate schedule with linear warm-up | 5 min | lr at step 0 = 0, at warm-up end = peak, at end ≈ 0 |

## Tier 2 — your own stack (they will ask about what you claim to have built)

| # | Component | Budget | Done when |
|---|---|---|---|
| 11 | Flow-matching training step: sample t, x_t = (1−t)·x₀ + t·x₁, predict velocity, MSE to (x₁ − x₀) | 10 min | loss decreases on a toy 2-D target |
| 12 | Euler sampler for the above, N steps, fixed noise seed | 8 min | N=1 vs N=10 outputs differ; same seed gives same output (parity habit) |
| 13 | Action chunk executor: policy returns K actions every call, executor plays them at control rate, re-plans with inference delay d, prefix overwrite at the boundary | 20 min | the jump at the boundary is smaller with overwrite than without, on a synthetic chunk sequence |
| 14 | JEPA step: encoder, EMA target encoder, predictor conditioned on action, stop-gradient on the target, loss in latent space | 20 min | EMA weights move; target gets no grad (`requires_grad=False` or `.detach()`), check `grad is None` |
| 15 | SIGReg term: project embeddings onto M random unit directions, compare the 1-D sketch to N(0,1) (Epps–Pulley or a moment-matching stand-in), add to loss | 15 min | collapsed constant embeddings get a large penalty; Gaussian embeddings a small one |
| 16 | Copy-last and random-init baselines wired into the same evaluation function as the model | 8 min | one `evaluate(fn)` call works for all three |
| 17 | Episode-level bootstrap CI for a success rate and for a paired difference | 10 min | resamples episodes, not frames; interval widens with fewer episodes |
| 18 | Wilson interval for k/n | 5 min | 7/10 → about (0.40, 0.89) |
| 19 | AUROC by hand (rank-based) | 10 min | matches `sklearn.metrics.roc_auc_score` |
| 20 | Dataset/DataLoader for episodic data: sample (obs_t, action_{t:t+K}) windows, pad the tail, return a mask; split by episode not by frame | 15 min | no episode appears in both splits; mask zero on padding |
| 21 | Normalisation stats computed on the train split only, applied at inference, saved and loaded | 8 min | leakage test: changing the val split does not change the stats |
| 22 | Strict checkpoint load: `load_state_dict(strict=True)`, count tensors loaded, assert equality with the paper's count | 5 min | a missing key raises; the count is printed |
| 23 | **Latent-space MPC with CEM**: encode obs and goal; sample N action sequences of horizon H from a Gaussian; roll out the predictor autoregressively; cost = ‖ẑ_H − z_goal‖²; keep top-k elites; refit mean/std; iterate; execute the first action(s), replan | 20 min | on a toy linear "predictor" z' = Az + Bu the planner drives z to the goal; cost of the elite mean decreases every iteration |
| 24 | **Distributed training skeleton (discussion, not cold code)**: DDP init and `DistributedSampler`, per-rank seeds, gradient accumulation with `no_sync`, bf16 autocast, checkpoint/resume (model, optimiser, scheduler, sampler epoch, RNG states), rank-0 logging | 10 min to sketch on a whiteboard | you can say what breaks going 1 → 8 GPUs (effective batch and lr, sampler epoch, BN stats, eval on rank 0 only, non-determinism) |
| 25 | **Discounted returns in one reverse pass** (G_t = r_t + γ·G_{t+1}, with episode-done resets), then GAE as the follow-up | 10 min | matches a naive O(T²) double loop on a random reward vector; resets at episode boundaries |
| 26 | **Data-modelling scaffold drill**: take a public tabular or small image dataset, write a deliberately mediocre training script with two planted issues (a leak and an imbalance), swap with a friend, find and fix in 45 min, justify each change | 45 min | each change comes with a sentence saying why it should help and a number showing it did |

## Tier 3 — algorithmic insurance (shrunk: the JD is research engineering, not generic SWE)

Only these, 15 minutes each, once: hash-map lookup (two-sum); BFS on a grid; top-k with a heap; in-place matrix ops with complexity stated. Skip linked lists, tries and LRU unless the recruiter says the loop has an algorithmic round; ask at the screen.

## The debugging drill (the "broken training script" round)

Rules from real debugging rounds: ask what is in scope in the first minute; state a hypothesis about where the bug lives **before** reading the code; test it with one targeted print or assert; five minutes per hypothesis, then the next. Do not audit the whole file.

Write one correct tiny training script (item 09), then make nine copies, each with one planted bug, and fix each copy cold in under 8 minutes using this fixed order: shapes and dtypes at every boundary → overfit one batch → one hand-computed example through the data pipeline → `eval()`, normalisation, `detach` → hyper-parameters last.

Bugs to plant: lr ×10 too high; `zero_grad` missing; `model.eval()` never called; target not detached (collapse); normalisation stats from the wrong split; loader `shuffle=False` on sorted data; off-by-one so the action target includes the current action; fp16 cast in the loss; a leaked feature correlated with the label (REAP v1's elapsed time).

## Schedule (reweighted for the World Models RE posting)

**Core, weeks 1–2:** 14 (JEPA step: EMA, stop-grad, collapse), 15 (SIGReg), 23 (latent CEM/MPC), 11–13 (flow matching, Euler, chunk executor), alongside tier 1 items 01–03 and 08. Expect to be asked to write or debug a JEPA step by name.
Week 1: 01, 02, 03, 14, 15, 23 — each twice (second time from memory the next day).
Week 2: 11, 12, 13, 08, 04–07, 09–10; item 24 as a whiteboard sketch.
Week 3: the debugging drill (the JD's "debug and scale training runs, keep experiments reproducible"), plus 16–22.
Week 4: mock round — someone picks three items at random from 01–03, 11–15, 23, 45 minutes, camera on, no notes. Tier 3 only if the recruiter confirms an algorithmic round.

## If the Data role (Research Scientist/Engineer, Data)

A different loop: experimental design and paper discussion, little cold coding. Keep 17, 20, 21 (bootstrap CI, episodic dataset with episode split, train-only statistics) and the debugging drill. Prepare instead, as one-page written answers you can defend:

1. **A data-mixture ablation under fixed compute.** Hold steps, batch, model and seeds fixed; vary only the mixture (e.g. teleop : rollouts : human video); a grid or a mixture-law fit over 4–6 points; at least 3 seeds per point (the 26-pp seed floor of 2610.19375); report on a held-out real-robot block, paired layouts.
2. **"Which data helps" with real-robot success rates.** Random-at-the-same-budget baseline; length baseline; the curation score as a judge first (AUROC against outcomes) and as a training weight second; matched continued-training control; power at the n a robot day affords (book §24, 13b, chapter 7).
3. **A scaling study you can afford.** 60 / 120 / 240 / 480 successes, same budget, seeds; where the curve should bend; how you would know you are data-limited vs model-limited.
4. **Human video → robot actions.** Hand pose → end-effector → joints via IK; viewpoint alignment; latent actions when no labels exist; LaLa-World's language-as-action interface as a calibration-agnostic alternative (12b.8a); what you would measure first (book 16.6, §25).

