"""W5 D1: CLIP / SigLIP losses and a tiny CLIP."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """img_emb, txt_emb (N, d) NOT yet normalized; logit_scale = log of the scale (scalar).
    L2-normalize, logits = exp(logit_scale) * z @ u.T, symmetric cross-entropy (mean of both directions)."""
    # TODO
    raise NotImplementedError


def siglip_loss(img_emb: torch.Tensor, txt_emb: torch.Tensor, log_t: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalize, logits = exp(log_t) * z @ u.T + b, labels +1 diagonal / -1 elsewhere,
    loss = -mean(logsigmoid(labels * logits)) over all N*N pairs."""
    # TODO
    raise NotImplementedError


class TinyCLIP(nn.Module):
    """Image tower: Conv(3,32,3,p1) ReLU MaxPool2 Conv(32,64,3,p1) ReLU AdaptiveAvgPool(2) Flatten Linear(256, d).
    Text tower: nn.Embedding(9, d) summed over the 3 tokens, then Linear(d, d).
    logit_scale parameter initialised to log(10)."""

    def __init__(self, d: int = 32):
        super().__init__()
        # TODO

    def encode_image(self, img):
        # TODO
        raise NotImplementedError

    def encode_text(self, tokens):
        # TODO
        raise NotImplementedError


def train_clip(imgs, caps, steps: int = 1500, batch: int = 64, lr: float = 3e-3, seed: int = 0) -> TinyCLIP:
    # TODO
    raise NotImplementedError


@torch.no_grad()
def zero_shot_accuracy(model: TinyCLIP, imgs: torch.Tensor, caps: torch.Tensor, all_caps: torch.Tensor) -> float:
    """For each image, pick the caption among all_caps (K, 3) with the highest cosine similarity.
    Accuracy = fraction where it equals the true caption."""
    # TODO
    raise NotImplementedError
