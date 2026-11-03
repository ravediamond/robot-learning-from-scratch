# Predict BEFORE running `predict_run.py`

1. LoRA r = 8 on a 768 × 768 Linear: trainable parameters →   vs full →
2. LoRA r = 16 on q_proj and v_proj of all 12 layers of a D = 768 model: trainable →
3. Adam memory for full fine-tuning of a 450M VLA (fp32 states) →   with LoRA (10M trainable) →
4. Output of a freshly wrapped LoRA layer vs the original: equal or not? →

## Misses
-
