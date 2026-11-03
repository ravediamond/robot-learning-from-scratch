"""REVIEW. Resume logic of a training job that is preempted every few hours. After each resume
the loss jumps up for a while, and the final model is worse than an uninterrupted run.
Find the bug, fix, replace:
BUG:
"""
import torch


def save(path, model, opt, step):
    torch.save({"model": model.state_dict(), "step": step}, path)


def resume(path, model, opt):
    ck = torch.load(path, weights_only=True)
    model.load_state_dict(ck["model"])
    return ck["step"]
