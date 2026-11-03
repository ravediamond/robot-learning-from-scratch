# Follow-up answers (short)

m1: Scores are B·h·T² floats: at T = 8k, 64M per head-batch: GBs; use FlashAttention / tiling. Padding mask: (B, 1, 1, T) key mask ANDed with causal. RoPE is applied to q and k after the projection, before the dot product.
m2: At t = 0 the input carries no information about the target, the L2-optimal velocity points to the mean, one step lands there. Faster: fewer steps, distillation (consistency/shortcut), cache the conditioning prefix. DDPM: predict noise with a variance schedule and an ancestral or DDIM sampler.
m3: Summing rewards getting close early and penalises overshoot/oscillation; final-state-only can choose detours. N = 20 is too few elites to estimate a mean and std in H·A dims: noisy, premature convergence. Warm start: shift the previous mean by one step and pad, keep a smaller std.
m4: Paired removes start-difficulty variance. McNemar is exact for binary paired outcomes and needs no resampling; bootstrap gives an interval on the difference. For a 10-point gain: depends on the discordant rate; with ~25% discordant pairs, roughly 200+ starts for 80% power: say it's expensive and propose a denser metric.
