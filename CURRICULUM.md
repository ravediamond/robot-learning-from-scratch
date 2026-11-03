# Curriculum: 9 weeks, 45 sessions (all written)

Each lab: README (concept) → predict.md / predict_run.py → build with tests → review (planted bug) →
questions.md (out loud). Solutions and model answers: `solutions` branch. After each Friday, write the
retro in PROGRESS.md: labs get adjusted (easier/harder) from it.

From week 2 on, some `predict_run.py` scripts use YOUR code: run them after your tests pass.
K = knowledge primer to read that week (2h days or weekend).

## Week 1: PyTorch from zero
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Tensors & shapes | to_tensor, time flatten, padding + masks, action chunks with `unfold` | per-joint mean over the wrong dim |
| Tue | d2 Indexing, broadcasting, einsum | pairwise distances, `gather`, masked mean, best-of-N | masked loss divided by padded size |
| Wed | d3 Autograd | linear grads by hand, finite differences, GD loop, stop-gradient | gradients never zeroed |
| Thu | d4 nn.Module & training loop | MLP, train/eval, overfit one batch | validation with dropout on |
| Fri | d5 BC on a toy SO-101 | episode split, train-only stats as buffers, beat hold-position, strict save/load | frame-level split leaks episodes |

## Week 2: Training properly, CNN encoders, your real SO-101 data
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Datasets & loaders | episode windows with padding/mask, collate, seeded loader, task-balanced weights | windows cross episode boundaries |
| Tue | d2 Training stability | step-0 loss, warmup + cosine, grad clipping by hand, first non-finite module | clipping before backward |
| Wed | d3 CNN encoders | conv via unfold, receptive field, spatial softmax, GroupNorm encoder, random shift | HWC reshaped as CHW |
| Thu | d4 **Real data** (`ravediamond/so101_bowls_all_trimmed`) | read LeRobot parquet by hand, episodes, gaps, idle starts, action→state lag; `inspect_real.py` on your Mac | rows not sorted in time |
| Fri | d5 **Pixel + state policy** | chunk policy with CNN + state, beat state-only on toy pixels; `extract_frames.py` + `train_real.py` on your data | augmentation at evaluation |

## Week 3: Attention, transformers, GPT-2  · K: `01-transformers-attention`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Attention | SDPA by hand, causal/padding masks, fully-masked rows | causal mask hides the diagonal |
| Tue | d2 Multi-head & cross | MHA matching `nn.MultiheadAttention`, cross-attention | heads split along time |
| Wed | d3 Block & positions | pre-norm block, sinusoidal, RoPE, ViT patchify | missing residual |
| Thu | d4 **GPT-2 from scratch** | 124,439,808 params, HF names, tying, init | tied weights cloned |
| Fri | d5 **Load OpenAI's GPT-2** | HF weight loading (Conv1D transposes, strict count), generation, char model on your book | temperature on probabilities |

## Week 4: Generative models  · K: `04-diffusion-flow-matching`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 MSE fails, VAE | mean optimum, reparameterisation, KL, VAE covers both modes | exp(logvar) used as std |
| Tue | d2 DDPM | schedule, q_sample, ε-loss, ancestral sampler on two moons | noise coefficients swapped |
| Wed | d3 **Flow matching** | interpolation, velocity net, Euler sampler; 1 step collapses to the mean | time convention flipped |
| Thu | d4 Conditional flow + CFG | null label, label dropout, guided velocity | guidance weights swapped |
| Fri | d5 **Flow policy vs MSE, closed loop** | rollout loop in the bimodal obstacle world, collisions with Wilson CIs | actions not un-normalised/clipped |

## Week 5: VLMs and VLAs  · K: `02-vlm`, `03-vla`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 CLIP / SigLIP | both losses, tiny CLIP, zero-shot accuracy | embeddings not normalised |
| Tue | d2 VLM plumbing | pixel shuffle, projector, π0/SmolVLA block masks, prefix assembly | token merge without permute |
| Wed | d3 Action tokenizers | uniform bins, DCT (FAST-style) | decode to bin edges |
| Thu | d4 LoRA | LoRALinear, merge, apply to named modules | A and B both zero |
| Fri | d5 **Mini VLA** | patches + instruction + state prefix (cached), cross/causal action expert, flow; swapped-instruction test | time index instead of t in [0, 1) |

## Week 6: Robot learning and evaluation  · K: `07-robot-eval-statistics`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 ACT | CVAE chunk policy, L1 + KL, temporal ensembling; β/seed sweep | stale actions in the ensembler |
| Tue | d2 Statistics | Wilson, episode bootstrap, paired diff, McNemar, trials needed | frame-level bootstrap |
| Wed | d3 Failure monitors | AUROC with ties, CI, TPR@FPR, conformal threshold; `length_null_real.py` on your rollouts | ties ranked by row order |
| Thu | d4 **Closed-loop evaluator** | fixed paired starts, per-episode outcomes; val MSE ranks heads wrongly | starts change every call |
| Fri | d5 **DAgger** | pushes off the demo path, expert relabelling; BC vs DAgger paired | policy labels its own states |

## Week 7: JEPA and world models  · K: `05-jepa-ssl`, `06-world-models-planning`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Collapse & VICReg | VICReg terms, effective rank; watch invariance-only collapse | variance over the wrong dim |
| Tue | d2 SIGReg, EMA, stop-grad | Epps–Pulley, SIGReg, EMA update, stop-gradient loss | EMA weights swapped |
| Wed | d3 Probes & baselines | ridge and MLP probes, R², copy-last | R² normalised by predictions |
| Thu | d4 **Mini LeWM from pixels** | encoder + residual predictor + SIGReg; collapse at λ=0; beats copy-last, MLP probe > 0.9 | target is the same frame |
| Fri | d5 **Planning & surprise** | latent CEM: latent-L2 vs probe-decoded cost; surprise AUROC on teleports | CEM refits on all samples |

## Week 8: Robotics basics and your own register
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Kinematics | rotations, transforms, SO-101-like FK, Jacobian, damped least-squares IK | transposed Ry |
| Tue | d2 Cameras & calibration | projection, back-projection, Kabsch hand-eye fit | divide by distance, not depth |
| Wed | d3 Control & latency | PID, delay oscillation, chunk switching: naive / shift / blend (RTC idea) | derivative × dt |
| Thu | d4 **Re-derive your numbers** | your `counts_ci.json` Wilson audit, your `anchor_predictions.csv` AUROCs + CIs | seed-major CSV reshaped |
| Fri | d5 **Checkpoint forensics** | load report, key remapping, safe load, fraction changed; your W1 load audit | `strict=False` result ignored |

## Week 9: Mock interviews  · K: `08-question-bank`
| Day | Lab | Format |
|---|---|---|
| Mon | d1 DDP + exact resume | 2 CPU processes (gloo) == single process; bit-identical resume (review: optimizer state not saved) |
| Tue | d2 Timed coding | 45 min blank files: attention, flow loss + sampler, CEM, paired bootstrap + Wilson |
| Wed | d3 Fix the script | a BC training script with 7 planted bugs, 45 min, one hidden test per bug |
| Thu | d4 Research verdict | paper → experiment template (pre-registered), your pass / fail / not-proven rule as code |
| Fri | d5 System design & pushback | data-flywheel design (rubric), 5 pushback prompts, opener, final retro |

## LeetCode drills (15 min, any day): `drills/w1` … `drills/w8`
w1 hash map, stack, monotonic deque, intervals, binary search · w2 two pointers, sliding window ·
w3 heaps · w4 grid BFS/DFS · w5 union-find, topo sort, Dijkstra, cycles · w6 DP · w7 DP + scheduling ·
w8 design (LRU, trie, ring buffer, in-place rotation). Pattern notes: `drills/PATTERNS.md`.

## On your Mac only (needs internet or your data)
W2 D4 `download_bowls.py` + `inspect_real.py` · W2 D5 `extract_frames.py` + `train_real.py` ·
W3 D5 `GPT2_REAL=1` test + `train_char.py` · W6 D3 `length_null_real.py`.
