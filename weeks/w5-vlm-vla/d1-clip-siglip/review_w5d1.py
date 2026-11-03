"""REVIEW. SOLUTION.

BUG: the embeddings are not L2-normalized, so logits scale with the embedding norms: the encoders can
inflate norms to sharpen the softmax (and blow up the loss when wrong). Normalize, then apply the scale.
"""
import torch
import torch.nn.functional as F


def contrastive_loss(img_emb, txt_emb, scale=10.0):
    logits = scale * F.normalize(img_emb, dim=-1) @ F.normalize(txt_emb, dim=-1).T
    labels = torch.arange(len(img_emb))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
