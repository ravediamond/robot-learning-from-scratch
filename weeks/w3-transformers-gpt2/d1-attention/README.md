# W3 D1: Scaled dot-product attention, by hand

Book: P2.15, P2.20. Primer: `knowledge/01-transformers-attention.md`. Coding list item 01.

## The one idea
```
scores = Q Kᵀ / √d          (B, Tq, Tk)    how much each query matches each key
scores[mask == False] = −inf                 forbidden pairs
A = softmax(scores, dim=-1)                  rows sum to 1
out = A V                    (B, Tq, dv)
```
Masks are booleans "may attend" (True) with shape broadcastable to (B, Tq, Tk):
- **causal** (T, T): lower triangle including the diagonal. Token t sees 0..t.
- **key padding** (B, Tk): False on padded keys; expand to (B, 1, Tk).
- Combine with `&`.

A row with NO allowed key gives softmax over all −inf = NaN. Real code must decide what
that row means (here: output zeros).

PyTorch reference: `F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)` (True = attend).

## Today's build (`w3d1_attention.py`)
`attention`, `causal_mask`, `key_padding_mask`, `combine_masks`, plus `attention_weights`
returning A so you can look at it.

Run: `uv run pytest weeks/w3-transformers-gpt2/d1-attention`
