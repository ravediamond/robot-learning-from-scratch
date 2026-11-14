# Predict BEFORE running `predict_run.py`

1. In the training data the "gripper closed" flag is 1 in 95% of success states and 5% of the others. Held-out accuracy of the classifier ≈ →
2. Same classifier on states that are FAR from the goal but with the flag set to 1: fraction called success at threshold 0.5 →
3. Threshold chosen for precision ≥ 0.99 on validation: recall ≈ →
4. Progress model trained only on ORDER inside expert episodes. Spearman correlation with −distance on new episodes ≈ →
5. Same model, pairs built ACROSS episodes (the review bug): correlation ≈ →

## Misses
-
