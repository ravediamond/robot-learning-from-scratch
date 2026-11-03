# W5 D1: CLIP and SigLIP from scratch

Primer: `knowledge/02-vlm.md` (step 1). The vision tower of every VLA was trained this way.

## Contrastive losses
```
z = normalize(image_encoder(img)),  u = normalize(text_encoder(txt))     (N, d) each, unit norm
S = (z @ u.T) * scale                                                      (N, N), row i = image i vs all texts
CLIP:   loss = ( CE(S, arange(N)) + CE(S.T, arange(N)) ) / 2              softmax over the batch, both ways
SigLIP: labels = +1 on the diagonal, -1 elsewhere;  logits = z@u.T * t + b
        loss = - mean over ALL N*N pairs of logsigmoid(labels * logits)   (the paper sums over pairs / N; we take the mean)
```
`scale` (CLIP) and `t, b` (SigLIP) are learnable (stored as log for the scale).

**False negatives**: if two images in a batch have the same caption, CLIP treats the other one as a
negative. With only 24 distinct captions today, that happens in every batch. Watch what it does.

## Today's build (`w5d1_clip.py`)
`clip_loss`, `siglip_loss`, `TinyCLIP` (CNN image tower + bag-of-words text tower),
`train_clip`, `zero_shot_accuracy` (classify each image among the 24 captions).
Test: zero-shot accuracy > 90% on held-out images.

Run: `uv run pytest weeks/w5-vlm-vla/d1-clip-siglip`
