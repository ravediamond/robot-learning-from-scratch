# Curriculum: 8 weeks, 40 sessions

Each week ends with a Friday lab that ties the week to a robot. Labs for week N+1
are written after week N is done, so the level adjusts to how it actually went.

Legend: **Build** = graded exercise · **Review** = planted bug · **K** = knowledge primer to read that week (2h days or weekend)

## Week 1: PyTorch from zero  ✅ written
| Day | Lab | Build | Review bug (real failure type) |
|---|---|---|---|
| Mon | d1 Tensors & shapes | to_tensor, time flatten, padding + masks, action chunks with `unfold` | per-joint mean over the wrong dim |
| Tue | d2 Indexing, broadcasting, einsum | pairwise distances, `gather`, masked mean, best-of-N selection | masked mean divides by T, not by valid steps |
| Wed | d3 Autograd | gradients by hand vs autograd, finite-difference check, GD loop | gradients accumulate (no `zero_grad`) |
| Thu | d4 nn.Module & training loop | MLP, train/eval loops, overfit one batch, param count | eval without `model.eval()` / `no_grad` |
| Fri | d5 Behavior cloning on a toy SO-101 | episode split, train-only norm stats, normalization as buffers, beat the cheapest baseline, save/load round trip | frame-level split leaks episodes |

## Week 2: Training properly + CNN encoders
Data pipelines (Dataset/DataLoader, collate, padding) · initialization and why loss at step 0 should be ln(K) or Var(y) · LR finder and warmup · gradient clipping · NaN hunting · CNN encoder from scratch, receptive field · BatchNorm vs LayerNorm vs GroupNorm (and why policies avoid BatchNorm) · image augmentation for policies (random shift).
Friday: train a pixel → action encoder on rendered toy frames.

## Week 3: Attention and transformers  · K: `01-transformers-attention`
Scaled dot-product attention by hand · causal and padding masks · multi-head · transformer block (pre-norm, residual, MLP) · positional encodings (sinusoidal, learned, RoPE) · ViT patchify · cross-attention · KV cache.
Friday: a tiny transformer policy over a history of states.

## Week 4: Generative models  · K: `04-diffusion-flow-matching`
Why MSE fails on multimodal actions · VAE (reparameterization, KL) · DDPM forward/reverse, noise schedule · flow matching (linear path, velocity target, Euler sampling) · classifier-free guidance.
Friday: a flow-matching action head on a 2-mode toy task, MSE head as the baseline.

## Week 5: VLMs and VLAs  · K: `02-vlm`, `03-vla`
CLIP contrastive loss (InfoNCE) from scratch · image tokens into a language model (projector, pixel shuffle) · VLA anatomy: VLM backbone + action expert (SmolVLA layout) · action tokenization (binning, FAST) vs continuous heads · action chunking, temporal ensembling, real-time chunking · freezing vs fine-tuning, LoRA.
Friday: a mini VLA: frozen toy encoder + text embedding + flow-matching action expert with cross-attention.

## Week 6: Robot learning and evaluation  · K: `07-robot-eval-statistics`
Compounding error and DAgger · ACT (CVAE + chunks) · state/action normalization pitfalls · success-rate statistics: binomial CI, episode bootstrap, paired comparisons, how many trials you need · failure detection (score thresholds, conformal calibration) · reward models from progress.
Friday: evaluate two policies honestly; decide which is better with a pre-registered bar.

## Week 7: JEPA and world models  · K: `05-jepa-ssl`, `06-world-models-planning`
Representation collapse and why it happens · VICReg, SIGReg-style Gaussian regularizer, EMA target encoders · masked latent prediction (I-JEPA / V-JEPA) · action-conditioned latent dynamics (LeWM-mini) · CEM / MPPI planning in latent space · world model as evaluator: surprise, failure prediction · RSSM in one page.
Friday: train a LeWM-mini on toy pixels, plan to a goal with CEM, then measure surprise on perturbed rollouts.

## Week 8: Robotics basics, your own register, mock interviews  · K: `08-question-bank`
Mon: FK/IK for a 5-DoF arm (SO-101), Jacobians, frames · Tue: cameras, intrinsics/extrinsics, hand-eye calibration, control rate and latency · Wed: your register, re-derived by hand (gate verdicts, CIs, the checkpoint-loading bug, why best-of-N reranking failed) · Thu: timed mock coding (45 min: attention or flow-matching loss from a blank file) + "fix this training script" · Fri: research discussion mock (paper → experiment → verdict) + system design (a data flywheel on edge hardware).

## LeetCode drills (15 min, any day)
Week 1: hash map, stack, sliding window + deque, intervals, binary search.
Next weeks: two pointers, heaps / top-k, BFS/DFS on grids, union-find, DP basics, graphs (Dijkstra = path planning), tries, intervals scheduling.
