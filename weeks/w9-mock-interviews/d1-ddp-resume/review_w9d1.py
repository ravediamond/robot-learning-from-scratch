"""REVIEW. SOLUTION.

BUG: the optimizer state is not saved or restored. Adam restarts with zero moments (and bias correction at
step 1): huge effective steps right after each resume -> loss spikes; with a scheduler, the lr also restarts.
Save and load opt.state_dict() (and the scheduler, step, RNG).
"""
import torch


def save(path, model, opt, step):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, path)


def resume(path, model, opt):
    ck = torch.load(path, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return ck["step"]
