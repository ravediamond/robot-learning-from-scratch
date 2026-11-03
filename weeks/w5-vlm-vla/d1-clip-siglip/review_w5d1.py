"""REVIEW. A contrastive loss for aligning robot camera frames with instructions. Training is
unstable: the loss swings wildly and depends on the scale of the encoders' outputs.
Find the bug, fix, replace:
BUG:
"""
import torch
import torch.nn.functional as F


def contrastive_loss(img_emb, txt_emb, scale: float = 10.0):
    logits = scale * img_emb @ txt_emb.T
    labels = torch.arange(len(img_emb))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
