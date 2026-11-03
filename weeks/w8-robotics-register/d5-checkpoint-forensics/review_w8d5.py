"""REVIEW. SOLUTION.

BUG: strict=False with the result ignored: when the keys don't match (a "model." prefix here) NOTHING is
loaded, the model stays random, and the log still says "loaded". Load strictly (or check missing /
unexpected keys and the count), and fail loudly.
"""
import torch


def load_pretrained(model, path):
    sd = torch.load(path, weights_only=True)
    res = model.load_state_dict(sd, strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing {res.missing_keys[:5]}, unexpected {res.unexpected_keys[:5]}")
    print(f"loaded {len(sd)} tensors from {path}")
    return model
