# W5 D3: Action tokenizers: bins vs DCT (FAST-style)

Primer: `knowledge/03-vla.md`. RT-2 / OpenVLA emit actions as text tokens; π0-FAST compresses
chunks with a DCT first. Knowing why decides "tokens or flow head" in a design discussion.

## Uniform binning (RT-2, OpenVLA)
Each action dimension, each time step → one of n bins between [low, high]. A chunk of H×D values →
H×D tokens. Decode to the bin CENTRE: max error = half a bin width. For H = 50, D = 6: 300 tokens,
generated autoregressively: slow, and consecutive tokens are highly redundant (smooth motion).

## DCT compression (the core of FAST)
Smooth signals have their energy in the low-frequency coefficients of a Discrete Cosine Transform.
```
C (H x H) orthonormal DCT-II:  C[k, n] = s_k cos(pi (n + 0.5) k / H),  s_0 = sqrt(1/H), s_k = sqrt(2/H)
coeff = C @ chunk                 (H, D)   orthonormal: chunk = C.T @ coeff
keep the first K rows, round(coeff * scale) -> integer tokens (K, D)
decode: chunk_hat = C[:K].T @ (tokens / scale)
```
FAST then runs BPE on those integers to merge common patterns; we stop at the DCT + quantization.

## Today's build (`w5d3_tokenizers.py`)
`bin_encode`, `bin_decode`, `dct_matrix`, `dct_encode`, `dct_decode`.
Test: on smooth chunks, the DCT reaches binning's accuracy with far fewer tokens.

Run: `uv run pytest weeks/w5-vlm-vla/d3-action-tokenizers`
