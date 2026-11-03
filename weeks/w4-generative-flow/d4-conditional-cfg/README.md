# W4 D4: Conditional flow and classifier-free guidance

Primer: `knowledge/04-diffusion-flow-matching.md` (CFG). Tomorrow's action head is conditional on
the observation; a VLA is conditional on images AND an instruction. Today: the condition is a class.

## Conditioning
`v(x, t, c)`: add a learned embedding of the condition c to the input. For CFG, reserve one extra
embedding index as **null** ("no condition").

## Classifier-free guidance
- Training: with probability p (≈ 0.1) replace c by null. One network learns both v(x,t,c) and v(x,t,∅).
- Sampling: `v = v_uncond + w * (v_cond - v_uncond)`.
  w = 0: unconditional; w = 1: plain conditional; w > 1: push further toward the condition.
- Cost: 2 forward passes per step.

## Today's build (`w4d4_cfg.py`)
`CondVelocityNet` (class embedding with a null index), `drop_labels`, `cond_fm_loss`,
`cfg_velocity`, `cfg_sample`, `train_cond_fm`. Data: 4 Gaussian blobs (`toy2d.clusters`).

Run: `uv run pytest weeks/w4-generative-flow/d4-conditional-cfg`
