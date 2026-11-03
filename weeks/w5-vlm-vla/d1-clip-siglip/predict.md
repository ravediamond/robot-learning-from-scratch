# Predict BEFORE running `predict_run.py`

1. CLIP loss at init with batch N = 64 (random embeddings, scale 1) ≈ →   with N = 1024 →
2. Batch of 64, 24 distinct captions: expected number of images that share a caption with at least one other image →
3. Perfectly trained model, but every batch has duplicates: can the CLIP loss reach 0? →
4. SigLIP loss at init (logits ≈ 0) ≈ →

## Misses
-
