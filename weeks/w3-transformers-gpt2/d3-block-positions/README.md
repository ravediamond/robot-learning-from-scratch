# W3 D3: The transformer block, positions, image patches

Primer: `knowledge/01-transformers-attention.md`. Book: P2.20.

## Pre-norm block
```python
x = x + attn(ln_1(x), mask)       # residual 1
x = x + mlp(ln_2(x))              # residual 2, mlp: D -> 4D -> GELU -> D
```
Property worth knowing: if the last layer of `attn` and of `mlp` output zero, the block is the
identity. Many models initialise those layers near zero so a deep stack starts as identity
(stable training). That property is also your test today.

## Positions
- **Sinusoidal** (original transformer): `PE[t, 2i] = sin(t / 10000^(2i/D))`, `PE[t, 2i+1] = cos(...)`. Added to the input.
- **Learned absolute** (GPT-2): an `nn.Embedding(max_T, D)`.
- **RoPE** (LLaMA, most modern VLMs): rotate each pair (x₂ᵢ, x₂ᵢ₊₁) of q and k by angle `t·θᵢ`,
  `θᵢ = base^(−2i/d)`. Then `q_m · k_n` depends only on `m − n`.
```
x2i'   = x2i·cos(tθ) − x2i+1·sin(tθ)
x2i+1' = x2i·sin(tθ) + x2i+1·cos(tθ)
```

## ViT patches
`(B, C, H, W)` → cut into `p×p` patches → `(B, N, C·p·p)` with `N = (H/p)·(W/p)`, patches in
row-major order, each patch flattened as (C, p, p). Then one Linear = "patch embedding".

## Today's build (`w3d3_block.py`)
`Block`, `sinusoidal_positions`, `apply_rope`, `patchify`.

Run: `uv run pytest weeks/w3-transformers-gpt2/d3-block-positions`
