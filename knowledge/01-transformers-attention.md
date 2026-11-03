# 01. Transformers and attention

## Mechanism

**Attention** = a differentiable lookup. Each query asks "which keys look like me?" and
takes a weighted average of their values.

```
Q = X Wq   K = X Wk   V = X Wv            X: (B, T, D)   Q, K, V: (B, T, d)
A = softmax(Q Kᵀ / √d + mask)             (B, T, T)   rows sum to 1
out = A V                                  (B, T, d)
```
- `/√d`: q·k is a sum of d terms; its variance grows with d. Without scaling the softmax
  saturates (one-hot), gradients vanish.
- **Causal mask**: −inf above the diagonal, so token t cannot see t+1.. (needed for
  autoregressive generation, and for SmolVLA's action tokens attending to each other causally).
- **Padding mask**: −inf on padded keys.
- **Multi-head**: split d into h heads of d/h, attend separately, concat, project. Each head
  can learn a different relation (position, object, gripper state) at the same cost.
- **Self vs cross attention**: self: Q, K, V from the same sequence. Cross: Q from one
  (action tokens), K/V from another (image + text features). VLA action experts use both.

**Transformer block (pre-norm)**
```
x = x + Attn(LN(x))
x = x + MLP(LN(x))            MLP: d -> 4d -> d, GELU/SwiGLU
```
Residuals give gradients a straight path; pre-norm (LN before the sublayer) trains stably
without warmup tricks at depth.

**Positions.** Attention is permutation-invariant: without position info, a sequence is a
bag. Options: sinusoidal (fixed), learned absolute, **RoPE** (rotate q and k by an angle
proportional to position, so q·k depends on relative offset; used by most LLMs and many VLMs).

**ViT.** Cut the image into 16×16 patches, flatten each, linear-project to d: a 224×224
image becomes 196 tokens. Add positions, run a transformer. No convolutions.

**Cost.** Attention is O(T²·d) time and O(T²) memory per head. A 4-camera, 2-frame VLA
input can be thousands of tokens: that is why VLMs compress visual tokens (pixel shuffle,
pooling) and why FlashAttention (tiling, no materialized T×T matrix) matters.

**KV cache.** At generation time, past keys/values do not change: cache them, compute
only the new token's q, k, v. Turns O(T²) per step into O(T). For a VLA: the prefix (image +
text) is encoded once per inference and cached; the action expert attends to it.

## Questions

1. Why divide by √d? What happens at d = 1024 without it?
<details><summary>Answer</summary>If q and k have unit-variance entries, q·k has variance d. At d = 1024 logits have std 32: the softmax is essentially one-hot, its gradient is ~0, and training stalls or becomes unstable. Dividing by √d brings the variance back to 1.</details>

2. Write the shapes for multi-head attention with B=2, T=10, D=64, h=8.
<details><summary>Answer</summary>Q,K,V after projection (2, 10, 64) → reshape (2, 10, 8, 8) → transpose (2, 8, 10, 8). Scores (2, 8, 10, 10). Out (2, 8, 10, 8) → transpose/reshape (2, 10, 64) → output projection (2, 10, 64).</details>

3. Self-attention vs cross-attention in a VLA action expert: which does what?
<details><summary>Answer</summary>Cross-attention: action tokens (queries) read the VLM's image/text/state features (keys/values): "what does the scene say I should do". Self-attention among action tokens: coordinate the chunk internally (smoothness, consistency across time steps). SmolVLA interleaves the two and reports that interleaving beats either alone (book 11a.1).</details>

4. Why pre-norm over post-norm?
<details><summary>Answer</summary>In pre-norm the residual stream is never normalized, so the identity path carries gradients cleanly through many layers; post-norm puts LN on the residual path, which makes deep stacks hard to train without careful warmup. Cost: pre-norm's final output needs its own LN.</details>

5. How does RoPE encode relative position?
<details><summary>Answer</summary>It rotates each 2-D pair of q and k dimensions by an angle mθ_i (m = position). The dot product of a rotated q at m and k at n depends only on (m − n). So attention scores are a function of relative offset, with no learned position table, and it extrapolates somewhat beyond training lengths.</details>

6. Your VLA input: 3 cameras × 256 tokens + 30 text + 1 state + 50 action tokens. Attention cost per layer relative to 1 camera?
<details><summary>Answer</summary>T goes from ~337 to ~849 tokens; cost scales with T², so ≈ 6.3×. Ways out: fewer visual tokens per camera (pixel shuffle to 64, pooling), cache the prefix, run the expensive VLM less often than the action expert.</details>

7. What is a KV cache, and does it help a VLA that outputs a whole chunk in one flow-matching pass?
<details><summary>Answer</summary>Cache of past keys/values so autoregressive decoding only computes the new token. For a flow-matching action expert, the prefix (image/text) is fixed across the ~10 denoising steps: caching its K/V means each step only processes the 50 action tokens. That is a large part of the speed of π0-style and SmolVLA inference.</details>

8. Interviewer: "Why not a CNN for the vision encoder?"
<details><summary>Answer</summary>CNNs have strong locality priors and are data-efficient; ViTs scale better with data and share one architecture with the language model, so pretrained VLM weights (trained on billions of image-text pairs) can be reused. On small robot datasets trained from scratch, CNNs (ResNet-18 in ACT/Diffusion Policy) are still competitive. The real reason VLAs use ViTs is pretraining, not the architecture itself.</details>
