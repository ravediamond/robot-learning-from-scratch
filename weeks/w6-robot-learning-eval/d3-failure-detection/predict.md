# Predict BEFORE running `predict_run.py`

1. A monitor that outputs the same score for every episode: AUROC →
2. A monitor whose scores are perfectly reversed (failures lowest): AUROC →   how would you fix it? →
3. 21 failures, 44 successes, true AUROC 0.8: width of the 95% CI, roughly (±0.02? ±0.1? ±0.2?) →
4. Conformal threshold from 19 calibration successes at α = 0.1: which order statistic (k-th smallest)? →  With only 5 calibration successes? →
5. On YOUR rollouts, guess the length-null AUROC before running length_null_real.py →

## Misses
-
