# Predict BEFORE running `predict_run.py`

1. HF GPT-2 `transformer.h.0.attn.c_attn.weight` shape for GPT-2 small →   your `h.0.attn.c_attn.weight` →
2. If you forget to transpose a SQUARE weight (c_attn.c_proj is 768×768), does loading fail? →
3. Greedy decoding (temperature 0) from the same prompt twice: same text? →   With temperature 1, top-k 50? →
4. Char-level model on your book: vocabulary size, roughly →   loss of a model that only knows character frequencies (unigram entropy, in nats) roughly →
5. Generating 200 tokens without a KV cache from a 20-token prompt: how many token-positions are processed in total? →

## Misses
-
