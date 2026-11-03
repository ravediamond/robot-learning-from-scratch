# W3 D5: Load OpenAI's GPT-2 into YOUR model, generate, train on your book (capstone, part 2)

Uses your `../d4-gpt2-model/w3d4_gpt2.py` (imported automatically).

## Loading someone else's weights: the job you will do for every VLA
1. Map names (theirs → yours). Here: drop the `transformer.` prefix.
2. Fix layouts. OpenAI's GPT-2 uses a `Conv1D` layer whose weight is stored as (in, out), the
   TRANSPOSE of `nn.Linear` (out, in). Four weights per block need `.t()`:
   `attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`.
3. Skip their non-parameter buffers (old versions store `attn.bias` / `attn.masked_bias` masks).
4. **Count**: every one of your tensors must be filled exactly once, shapes equal. A silent
   partial load gives a model that runs and produces garbage. (Your own register has a row about this.)
5. **Check outputs**: same input → same logits (to 1e-4). Only then trust it.

## Generation
```
for _ in range(n):
    logits = model(idx[:, -n_positions:])[0][:, -1] / temperature      # last position only
    keep top-k logits (others -inf), softmax, sample (or argmax if temperature == 0)
    idx = cat(idx, next)
```
Crop the context to `n_positions`: the model has no position embedding beyond it.
(A KV cache would avoid recomputing the prefix every step: see the questions.)

## Today's build (`w3d5_load_generate.py`)
`load_hf_state_dict`, `generate`, `CharTokenizer`, `get_batch`.

Tests use a small randomly initialised Hugging Face GPT-2 (no download). Then on your Mac:
```bash
GPT2_REAL=1 uv run pytest weeks/w3-transformers-gpt2/d5-gpt2-weights-generate -k real -s   # downloads 500 MB once
uv run python train_char.py      # ~10 min on MPS: a small GPT trained on YOUR book, then samples
```
