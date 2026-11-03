# Predict BEFORE running `predict_run.py`

Latent = [sin 3.8x, cos 3.8x, sin 3.8y, cos 3.8y] for positions (x, y) in [-0.8, 0.8]^2.
1. Linear probe R² for (x, y) →   MLP probe R² →
2. Probe fitted and evaluated on the SAME frames vs on held-out episodes: which is higher, and why can that mislead? →
3. Random features (a random linear map of the image) with a linear probe: R² roughly →
4. A latent with copy-last error 0.001 and predictor error 0.0009: good world model? →

## Misses
-
