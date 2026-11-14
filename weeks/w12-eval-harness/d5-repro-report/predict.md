# Predict BEFORE running `predict_run.py`

1. Two state dicts differ in one value in the middle of a 10,000-element tensor. Do their `str()` representations differ? →
2. Same weights saved as fp32 and as fp16. Same fingerprint? →
3. You change one hyperparameter in a nested config: `{"policy": {"chunk": 50}}` → 25. What does manifest_diff report? →
4. 31 successes out of 50. What does the table show (Wilson 95% CI, rounded percents)? →

## Misses
-
