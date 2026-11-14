# Predict BEFORE running `predict_run.py`

1. Episode lengths: 199 episodes between 240 and 420 frames, one of 6,000 frames. Mean ± 3σ outlier test: is the 6,000-frame episode flagged? Is the 45-frame one? →
2. Same with the median/MAD robust z-score (|z| > 3.5)? →
3. At 30 fps, a frozen run of 300 identical frame hashes is how many seconds? →
4. A joint stuck at 0.37 rad for a whole episode while the others move. Its range? Would a "values outside limits" check catch it? →
5. Two uploads of the same episode, renamed. Which check catches them, and is it cheap? →

## Misses
-
