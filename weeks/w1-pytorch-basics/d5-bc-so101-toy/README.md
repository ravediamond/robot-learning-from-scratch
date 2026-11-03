# W1 D5: Behavior cloning on a toy SO-101

Book: chapters 3 (the baseline is the method), 6 (trust the instrument), P2.8, P2.12.
Coding list items 16, 20, 21, 22. Your register: the checkpoint-loading audit
(`reference/journal/papers/jepa-world-model/AUDIT_W1_LOADS.md`).

## The one idea

Behavior cloning (BC) = supervised learning from (observation, action) pairs recorded by an
operator. Everything you did this week, applied to a robot. The ML is easy. The traps are
all in the **data handling** and the **evaluation**, and they are the same traps that bite
real policies:

1. **Split by episode, never by frame.** Frames 41 and 42 of one episode are near copies. A
   random frame split puts one in train and one in val: val loss looks great and means nothing.
2. **Normalization stats from the train split only**, then saved WITH the model. A policy
   loaded without its stats outputs garbage that looks like a "bad policy".
3. **Beat the cheapest baseline first.** Here: "hold position" (action = current state).
   If the model does not clearly beat it, nothing else matters.
4. **Save, reload, compare outputs.** If the reloaded model gives different numbers, every
   later evaluation is measuring something else. (You lived this one.)

## `register_buffer`

```python
self.register_buffer("obs_mean", mean)   # saved in state_dict, moved by .to(device), NOT trained
```
Normalization stats belong in buffers: they travel with the checkpoint and the device.

## Today's build (`w1d5_bc.py`)

1. `split_by_episode`
2. `stack_frames`: list of episodes → (N, 12), (N, 6)
3. `compute_norm_stats`
4. `BCPolicy`: normalizes inside `forward`, raw units in, raw units out
5. `train_bc`, loss in normalized action space
6. `hold_baseline_mse`
7. `save_policy` / `load_policy` with a strict load

The test trains for real (about 10–20 s on CPU) and requires val MSE < 75% of the
baseline's. When it passes, print both numbers and look at them: that is your first verdict.

Run: `uv run pytest weeks/w1-pytorch-basics/d5-bc-so101-toy`
