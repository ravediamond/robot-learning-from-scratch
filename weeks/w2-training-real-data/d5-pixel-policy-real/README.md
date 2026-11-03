# W2 D5: A pixel + state policy, on toy pixels then on your real data

Book: 11.1–11.2 (BC, chunking), 10b. Builds on D1 (windows), D3 (CNN + random shift), D4 (real data).

## The one idea
Most of what tells a policy WHERE to go is in the image, not in the joint state: where the bowl
is today. A policy that ignores the image can only learn the average trajectory. Today you
build a policy that fuses both, and you prove the image branch is actually used by beating a
state-only twin, first where you know the answer (toy) and then on your own data.

```
img (B,3,S,S) -> CNN (+ random shift in training) -> (B, 64) ┐
state (B, Ds) -> normalize -> Linear -> (B, 64)              ┴ concat -> MLP -> (B, H*Da) -> (B, H, Da)
```
Normalization stats (state, action) live in buffers, as in W1 D5. Loss: masked MSE over the
chunk in normalized action space.

## Today's build (`w2d5_pixel_policy.py`)
`masked_chunk_mse`, `PixelStatePolicy(use_image=...)`, `train_policy`, `evaluate`.
Test: on `dot_world.py`, pixel+state must reach < 30% of the state-only val MSE.

## On your Mac, after the tests pass (~20 min, once)
```bash
uv sync --extra video
uv run python ../d4-real-so101-data/download_bowls.py --videos
uv run python extract_frames.py ~/data/so101_bowls_all_trimmed --camera front_top_camera --size 64
uv run python train_real.py ~/data/so101_bowls_all_trimmed --episodes 60
```
Note the decoding speed `extract_frames.py` prints (frames/s): that is why loaders cache frames.
Write the three val numbers in `PROGRESS.md`, then answer the questions.
