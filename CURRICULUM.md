# Curriculum: Part 1 (9 weeks, 45 sessions) and Part 2 (weeks 10–14 and growing)

**Where the lessons are:** each week folder has a `README.md` with the full lesson for its five days (weeks 1–9;
weeks 10–14 point to their primer). Start at `weeks/w1-pytorch-basics/README.md`. This file is only the map.

Each lab: predict.md → the lesson (week page) → predict_run.py → build with tests → review (planted bug) →
questions.md (out loud). Solutions and model answers: `solutions` branch. After each Friday, write the
retro in PROGRESS.md: labs get adjusted (easier/harder) from it.

From week 2 on, some `predict_run.py` scripts use YOUR code: run them after your tests pass.
K = knowledge primer to read that week (2h days or weekend).

**Weekly reading slot (20 min, read only):** one real file per week, listed in `reading/README.md`
(LeRobot normaliser, dataset windows, SmolVLA expert and flow head, the train script; your eval code,
your LeWM, your RTC controller, your harness). Maps: `knowledge/09` (LeRobot), `knowledge/10` (Orin harness).

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

## Part 2 (see `ROADMAP.md`; for a data / evaluation / deployment application, `RE_TRACK.md` gives the order)

## Week 10: Reinforcement learning  · K: `11-reinforcement-learning`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Returns & policy gradient | returns, GAE, Gaussian policy, REINFORCE with a baseline | returns leak across episodes |
| Tue | d2 PPO | clipped objective, normalisation, update loop, solves Reach2D | `max` instead of `min` |
| Wed | d3 SAC | tanh-Gaussian log-prob, twin-Q target, soft update, temperature, replay | missing tanh correction |
| Thu | d4 Offline RL (IQL) | expectile value, AWR policy; beats BC on mixed data | expectile weights swapped |
| Fri | d5 RECAP-style conditioning | value, n-step advantages, per-task indicator, conditioned policy | one global threshold |

## Week 11: Reward models and simulation  · K: `11`, `12-simulation-sim2real`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Reward models | success classifier, precision-first threshold, pairwise progress model | pairs across episodes |
| Tue | d2 MuJoCo basics | substeps, PD + gravity compensation, Jacobian, damped least-squares IK (SO-101-like arm) | `int()` truncates substeps |

## Week 12: The evaluation harness (flagship engine)  · K: `13-evaluation-at-scale` · `flagship/`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Protocol | counterbalanced blocked schedule, blinding, record validation, drift | policies run back to back |
| Tue | d2 Several policies | McNemar per pair, Holm, rank probabilities, minimum detectable difference | bootstrap breaks pairing |
| Wed | d3 Failure taxonomy | Cohen's κ, per-cause rates with CIs, stage funnel | rates over failures only |
| Thu | d4 Latency on target | percentiles, deadlines, sync vs async chunk budgets, starvation | p99 from mean + σ |
| Fri | d5 Reproducibility + **design** | fingerprints, manifests, diffs, results table; evaluation program design | checkpoint hashed via `str()` |

## Week 13: Robot data engineering  · K: `14-robot-data-engineering`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Dataset QA | frozen cameras, dead joints, limits, robust outliers, duplicates, quarantine | mean ± 3σ masking |
| Tue | d2 Curation & splits | near-duplicates, group splits, dataset fingerprint, removal ledger | random episode split |
| Wed | d3 Mixtures & scaling | temperature sampling, per-item weights, power-law fit, seeded ablations | weights not divided by size |
| Thu | d4 Video at scale | storage and decode budgets, worker sharding, shuffle buffer | workers read the same shards |
| Fri | d5 Flywheel + **design** | intervention segments, review queue, collection planning; fleet data platform | policy actions as labels |

## Week 14: Deployment on the robot's computer  · K: `15-edge-deployment`
| Day | Lab | Build | Review bug |
|---|---|---|---|
| Mon | d1 Export & parity | deployable wrapper with stats as buffers, ONNX, parity, dynamic batch; TensorRT notes | batch frozen at 1 |
| Tue | d2 Quantisation | per-tensor vs per-channel int8, percentile calibration, fake quant + STE | no straight-through |
| Wed | d3 Real-time chunking | shift, soft mask, guided flow sampling; mode switches and seam jumps | prefix not shifted |
| Thu | d4 Safety layer | limits, rate limits, capture-time watchdog, NaN guard, e-stop latch | staleness from receipt time |
| Fri | d5 Fleet rollout + **design** | fingerprint check, shadow mode, canary rate test, stages; VLA on 20 robots | counts instead of rates |

## Live PyTorch rounds (30–45 min, AI off): `drills/live` L01–L10
causal MHA with KV cache · LayerNorm/RMSNorm · cross-entropy with ignore index and smoothing · top-k/top-p ·
RoPE · flow-matching loss + sampler · training step (accumulation, clipping, warmup + cosine) · transformer
block + parameter count · IoU + NMS · quaternions.

## LeetCode drills (15 min, any day): `drills/w1` … `drills/w8`
w1 hash map, stack, monotonic deque, intervals, binary search · w2 two pointers, sliding window ·
w3 heaps · w4 grid BFS/DFS · w5 union-find, topo sort, Dijkstra, cycles · w6 DP · w7 DP + scheduling ·
w8 design (LRU, trie, ring buffer, in-place rotation). Pattern notes: `drills/PATTERNS.md`.

## On your Mac only (needs internet or your data)
W2 D4 `download_bowls.py` + `inspect_real.py` · W2 D5 `extract_frames.py` + `train_real.py` ·
W3 D5 `GPT2_REAL=1` test + `train_char.py` · W6 D3 `length_null_real.py` · `flagship/` (your SO-101 + Orin).
