# W14 D2: quantisation you can defend

Primer: `knowledge/15-edge-deployment.md` §2.

## The one idea

int8 stores each weight in one byte: `w ≈ scale · q`, `q ∈ [−127, 127]`, `scale = max|w| / 127`
(symmetric). Two choices decide the error:

- **Granularity.** One scale for a whole matrix (**per-tensor**) is wrecked by a single large row:
  every other row is rounded on a coarse grid. One scale per output channel (**per-channel**) fixes it.
- **Calibration** of activations: their range is measured on real data. **Min-max** is ruined by rare
  outliers; a **percentile** (99.9%) clips them and keeps resolution for everything else.

Post-training quantisation (PTQ) does this after training. Quantisation-aware training (QAT) simulates
it during training with **fake quantisation**: quantise then dequantise in the forward pass. Rounding
has zero gradient almost everywhere, so the backward pass uses the **straight-through estimator**:
pretend the rounding was the identity (`x + (q − x).detach()`).

## Today's build (`w14d2_quant.py`)

`quantize`, `dequantize`, `percentile_scale`, `fake_quant`, `quantize_linear_weights`.

## Staff angle

The question is never "does int8 work" but "what does it do to the actions": report action error in joint
units and a short closed-loop check per quantised variant, and know which layers to keep in higher
precision (the first and last layers, small heads, normalisation).
