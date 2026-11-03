# W3 D2: Multi-head attention and cross-attention

Primer: `knowledge/01-transformers-attention.md`. Coding list item 02.

## Shapes, slowly
```
x (B, T, D) --q_proj--> (B, T, D) --view--> (B, T, h, D/h) --transpose(1,2)--> (B, h, T, D/h)
                    same for k, v
attention per head          -> (B, h, T, D/h)
transpose(1,2) + reshape    -> (B, T, D) --out_proj--> (B, T, D)
```
The `view` splits the LAST dim (features) into heads. Splitting any other way mixes time steps
across heads: shapes still fine, model silently wrong (today's review).

**Cross-attention**: queries from one sequence, keys/values from another:
`q = q_proj(x)`, `k = k_proj(context)`, `v = v_proj(context)`. In a VLA action expert: x = action
tokens, context = VLM features. Tq ≠ Tk is normal.

`nn.MultiheadAttention` stores q/k/v projections stacked in `in_proj_weight` (3D, D). Matching it
exactly is the test: if you can reproduce PyTorch's numbers, your shapes are right.
Careful: its `key_padding_mask` uses True = IGNORE (the opposite of yesterday's convention).

## Today's build (`w3d2_mha.py`)
`MultiHeadAttention(D, h)` with `q_proj, k_proj, v_proj, out_proj` (all `nn.Linear(D, D)`),
`forward(x, context=None, mask=None)`; `split_heads`, `merge_heads`.
You may import yesterday's ideas but write `attention` again inline: it is 4 lines.

Run: `uv run pytest weeks/w3-transformers-gpt2/d2-multihead-cross`
