# Questions (out loud). This is the Friday "connect it to a robot" set.

1. Why split by episode? Give the frame-split number from `predict_run.py` as your evidence, in one sentence an interviewer will remember.
2. Your BC model beats hold-position on val MSE. Does that mean it will succeed on the arm? Name three reasons it might not. (Compounding error is one: explain it in 30 s.)
3. Why predict absolute joint targets (LeRobot default) instead of deltas? When are deltas better?
4. Why compute the loss in normalized action space? What happens to the gripper dim if you compute MSE in raw units?
5. You load a checkpoint and the arm moves to a strange pose. List the checks, in order, that would separate "bad policy" from "bad loading". (Read your own audit in `reference/journal/papers/jepa-world-model/AUDIT_W1_LOADS.md` after answering.)
6. The operator data is multimodal: sometimes they go around an obstacle on the left, sometimes right. What does an MSE-trained policy do? Which week of this course fixes it, and how?
7. What is the cheapest baseline for a pick-and-place success rate, and why is "the policy beats the baseline on val MSE" a weaker claim than "beats it on success rate with a CI"? (Book chapter 3, 7.)
8. 50 demos, val loss noisy between epochs. How do you pick the checkpoint without fooling yourself?
9. BC lands around 40–65% of the hold baseline's MSE and more epochs do not help. Why is there a floor? Read `so101_toy.py`: name the two sources of irreducible error. What would you add to the observation to lower it? (Interview version: "your val loss plateaus, how do you know if it is the model or the data?")
