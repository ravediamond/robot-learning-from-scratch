# W5 D2: VLM plumbing: visual tokens, projector, attention blocks

Primers: `knowledge/02-vlm.md` (step 2), `03-vla.md`. Book: 11a.1 (SmolVLA).

## From image to tokens the language model can read
```
ViT patch features (B, N = h*w, C)
  -> pixel shuffle r:  merge each r x r block of neighbouring tokens into one token
                       (B, N / r^2, C * r^2)           SmolVLM: 1024 -> 64 tokens per frame
  -> projector MLP:    (B, N', C*r^2) -> (B, N', D_lm)
```
Pixel shuffle keeps all the numbers (moved into channels) but the language model now attends at a
coarser grid. Order inside a merged token: (row in block, column in block, channel).

## The attention mask of a VLA (π0 / SmolVLA style "blocks")
Sequence = `[ image + text tokens | state token | action tokens ]`.
- Block 1 (images + text): bidirectional inside, sees nothing after it.
- Block 2 (state): sees block 1 and itself.
- Block 3 (actions): sees everything before; inside itself, bidirectional (π0) or causal (SmolVLA's
  self-attention layers use a causal mask).
Why: the prefix (block 1) never depends on the actions, so it can be computed ONCE per inference and
cached while the action expert runs ~10 flow steps.

## Today's build (`w5d2_plumbing.py`)
`pixel_shuffle_tokens`, `Projector`, `block_attention_mask`, `assemble_prefix`.

Run: `uv run pytest weeks/w5-vlm-vla/d2-vlm-plumbing`
