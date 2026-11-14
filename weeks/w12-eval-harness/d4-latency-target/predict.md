# Predict BEFORE running `predict_run.py`

1. 95 samples at 10 ms and 5 at 200 ms. p50, p95, p99 (nearest rank)? Mean + 2.33σ? →
2. 30 Hz, chunks executed 25 actions at a time, inference 180 ms, synchronous. Idle fraction? →
3. Same, asynchronous, chunk length 50. Largest inference latency that never starves? →
4. Same but chunk length 30. Largest latency now? →
5. You measure 1,000 inferences including the first one (2.4 s: kernel compilation). Which statistics move? →

## Misses
-
