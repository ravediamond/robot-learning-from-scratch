# Fixes
1. split() split FRAMES: val frames had near-duplicates in train -> optimistic val loss (W1 D5).
2. norm_stats() ran on ALL demos before the split: val leaks into the statistics (W1 D5 / W2).
3. masked_loss() divided by all entries incl. padding: end-of-episode steps under-weighted (W1 D2).
4. lr_at(step // 100, ...): warmup lasted 100x too long, the schedule never reached its peak (W2 D2).
5. val_loss() ran with dropout on (no eval()): noisy validation, wrong checkpoint choice (W1 D4).
6. stats in a plain dict, not buffers: not saved; the reloaded policy uses dummy stats (W1 D5, W8 D5).
7. no opt.zero_grad(): gradients accumulated across steps (W1 D3).
