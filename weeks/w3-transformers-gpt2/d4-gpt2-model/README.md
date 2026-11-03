# W3 D4: GPT-2 from scratch (capstone, part 1)

Book: P2.20 (tiny GPT, worked). Primer 01. Tomorrow you load OpenAI's real weights into THIS
model, so today's names must match the original exactly.

## Architecture (GPT-2 small: L=12, D=768, h=12, V=50257, T_max=1024)
```
idx (B, T) -> wte(idx) + wpe(arange(T))                    (B, T, D)
for block in h:                                             pre-norm
    x = x + attn.c_proj( causal_attention( split(attn.c_attn(ln_1(x))) ) )
    x = x + mlp.c_proj( gelu_tanh( mlp.c_fc(ln_2(x)) ) )
x = ln_f(x)
logits = lm_head(x)                                         (B, T, V), lm_head.weight IS wte.weight
```
- `c_attn` = one `Linear(D, 3D)` producing q, k, v stacked: split the last dim into 3 chunks of D.
- GELU with the tanh approximation: `F.gelu(x, approximate="tanh")`.
- LayerNorm eps 1e-5. No dropout today.
- **Weight tying**: the output projection reuses the input embedding matrix. Saves 38.6M
  parameters (31% of the model). It must be the SAME tensor, not a copy.
- **Init**: all Linear and Embedding weights ~ N(0, 0.02), biases 0; the two `c_proj` per block
  ~ N(0, 0.02/√(2L)) because each block adds 2 residual contributions.

## Today's build (`w3d4_gpt2.py`)
`GPT2Config`, `CausalSelfAttention`, `MLP`, `GPT2Block`, `GPT2` with `forward(idx, targets=None)`
returning `(logits, loss)`; loss = cross-entropy over all positions, targets already shifted by
the caller, `ignore_index=-1`. You may use `F.scaled_dot_product_attention(..., is_causal=True)`
now: you wrote it by hand on Monday.

Pass: GPT-2 small has exactly **124,439,808** parameters.
