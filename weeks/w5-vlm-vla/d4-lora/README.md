# W5 D4: LoRA from scratch, and what to freeze in a VLA

Primer: `knowledge/02-vlm.md` (question 6). Your REKI work is the frozen-VLM case.

## Low-rank adaptation
Keep the pretrained weight W (out × in) frozen. Learn a low-rank update ΔW = B A with
A (r × in), B (out × r), r ≪ min(in, out):
```
y = x W^T + b + (alpha / r) * (x A^T) B^T
A ~ N(0, 1/r) small random,  B = 0   ->  at init the model is EXACTLY the pretrained one
```
Only A and B train: r·(in + out) parameters instead of in·out. After training, merge:
`W' = W + (alpha/r) B A`: zero inference cost.

Why B = 0 and A random (not both 0)? Gradient of B is (scaled) `(x Aᵀ)ᵀ · grad_out`: zero if A = 0.
Gradient of A involves B: zero if B = 0. Both zero → both gradients zero forever.

## Freezing strategies for a VLA (decide, then measure)
frozen VLM + trained action expert · LoRA on the LM + trained expert · full fine-tune (forgets
language) · vision tower frozen always (small data).

## Today's build (`w5d4_lora.py`)
`LoRALinear` (frozen base, A, B, scale, `merged()`), `apply_lora(model, names, r, alpha)`,
`trainable_params`.

Run: `uv run pytest weeks/w5-vlm-vla/d4-lora`
