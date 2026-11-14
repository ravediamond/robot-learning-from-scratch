"""REVIEW. SOLUTION.

BUG: str(state_dict) is the printed form of the tensors, and PyTorch abbreviates any tensor with more than
1,000 elements ("..."), showing only the first and last few values (and few digits). Two checkpoints that
differ anywhere else print, and hash, the same. Hash the names, dtypes, shapes and raw bytes instead.
"""
import hashlib

import torch


def weights_hash(state_dict: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state_dict):
        t = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        h.update(t.numpy().tobytes())
    return h.hexdigest()
