# Predict BEFORE running `predict_run.py`

The script loads 60 toy episodes and prints facts about them.

1. Total number of frames (episodes are 60 to 120 steps long) →
2. Std of shoulder_pan (joint 0) vs std of gripper (joint 5) over all frames: which is bigger, roughly how much? →
3. MSE in raw units of the "hold position" baseline (action = current state), over all frames. Order of magnitude: 0.1, 10, 1000? →
4. If you split 60 episodes by RANDOM FRAMES (80/20), what fraction of val frames have an episode-mate in train? →
5. Nearest-neighbor "policy" (copy the action of the closest train frame) evaluated on a FRAME split vs an EPISODE split: which val MSE is lower, by how much (×2? ×10?)? →

## Misses (one line each)
-
