# Predict BEFORE running `predict_run.py`

1. Chunk length 16. Inference takes 3 control steps; 5 actions of the previous chunk were executed when the request was sent. Which slice of the previous chunk is still "in the future" when the new one is computed? →
2. Two equally likely modes (up, down). The previous chunk went up. Naive async: probability that the new chunk goes down? →
3. With RTC guidance on the overlap, the same probability? →
4. Mean seam jump |new[0] − prev_shifted[0]| naive vs RTC: ratio roughly? →

## Misses
-
