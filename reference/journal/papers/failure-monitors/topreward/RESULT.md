# TOPReward vs elapsed time — result (WS-085; protocol `TOPREWARD_PROTOCOL.md`, frozen 6241570)

Qwen3-VL-8B @ 0c351dd, the paper's success score (z(s_top) + z(s_yn)), vast RTX 3090, 2.71 h, **$0.69**.
Scores: `out/` (not committed: 1.8 MB, regenerable), analysis `../results/topreward.json`. 0 scoring errors.

**ArmnetBench (n = 2,047, 515 successes, 7 policies):**

| score | pooled AUROC | within policy | 95 % CI (within) |
|---|---|---|---|
| TOPReward success score | 0.653 | **0.661** | 0.631–0.692 |
| — trajectory margin s_top only | 0.603 | 0.610 | 0.578–0.643 |
| — full-video Yes/No s_yn only | 0.672 | 0.678 | 0.648–0.709 |
| episode length | 0.938 | **0.942** | 0.927–0.956 |
| TOPReward, clip cut at 50 % of T | 0.658 | 0.667 | 0.637–0.698 |
| length, cut at 50 % of T | 0.838 | 0.851 | 0.829–0.873 |

Paired Δ within policy (TOPReward − length): −0.31 to −0.25 (full), −0.22 to −0.15 (50 % cut). **Reading (frozen
rule): TOPReward is below elapsed time, on full episodes and at the 50 % cut.** It is also below Robometer-4B's
pooled 0.818 (unpaired). Per policy 0.59 (SmolVLA) to 0.71 (MolmoAct2); per task 0.32 to 0.97.

**Own rollouts (n = 69, 44 / 25; descriptive, not in the decision):** TOPReward 0.825 (CI 0.70–0.93), length 0.56
(0.43–0.69), policy identity 0.87 (upper bound); non-TurboVLA subset 0.78 (8 failures). On operator-stopped runs, where
length is blind, the training-free judge carries signal; on benchmark runs that end by timeout, length beats it by 0.28.

**What it cannot say.** K, frames per query and the Yes/No wording are ours (the paper does not give them); one model
size (8B; the paper's 32B scored higher on its own benchmark); top view only.
