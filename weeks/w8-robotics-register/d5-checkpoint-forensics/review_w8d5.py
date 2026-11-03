"""REVIEW. Loads a pretrained policy before fine-tuning. "The fine-tune starts from the
pretrained weights" says the log. The fine-tuned policy is no better than training from scratch.
Find the bug, fix, replace:
BUG:
"""
import torch


def load_pretrained(model, path: str):
    sd = torch.load(path, weights_only=True)
    model.load_state_dict(sd, strict=False)
    print("loaded pretrained weights from", path)
    return model
