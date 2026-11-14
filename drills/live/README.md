# Live PyTorch rounds (30–45 min each, AI off)

Every research-engineer loop has one: a blank file, a shared screen, "implement causal multi-head
self-attention", "write the training step", "sample from these logits". Reported examples include live
multi-head self-attention with a causal mask (Mistral AI Engineer), a broken training script to fix, and
numerical coding on tensors. The week labs taught you the pieces with scaffolding. These ten prompts
remove it: one function or module per file, a docstring, hidden-in-plain-sight tests.

## Protocol (as in the real round)
1. Timer: 30 min (L01, L07, L08: 45 min). Read the docstring once. Say the shapes of every tensor out loud
   before typing, and the complexity.
2. Write it without running anything for the first 15 minutes. Then run `uv run pytest drills/live -k L01`.
3. While coding, narrate: "B, T, C; split C into H heads of size C/H; scores are B, H, T, T; mask the upper
   triangle with -inf before softmax...". Interviewers grade the narration as much as the code.
4. Over time: stop, read the solution on the `solutions` branch, redo from blank 48 hours later.
5. Log time and the first bug you hit in `PROGRESS.md`.

| File | Prompt | Lab it comes from |
|---|---|---|
| L01 | causal multi-head self-attention with a KV cache | W3 D1–D2 |
| L02 | LayerNorm and RMSNorm from scratch | W3 D3 |
| L03 | cross-entropy with ignore index and label smoothing | W3 D4 |
| L04 | top-k / top-p (nucleus) filtering and sampling | W3 D5 |
| L05 | rotary position embeddings (RoPE) | W3 D3 |
| L06 | flow-matching loss and Euler sampler | W4 D3 |
| L07 | the training step: accumulation, clipping, warmup + cosine | W2 D2 |
| L08 | a pre-norm transformer block and its parameter count | W3 D3–D4 |
| L09 | box IoU and non-maximum suppression, vectorised | perception basics |
| L10 | quaternions: multiply, rotate vectors, batched | W8 D1 |
