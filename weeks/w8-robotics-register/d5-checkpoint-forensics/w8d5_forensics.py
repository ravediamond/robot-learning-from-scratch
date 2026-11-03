"""W8 D5: SOLUTION."""
import torch
import torch.nn as nn


def load_report(model, sd):
    ours = model.state_dict()
    both = set(ours) & set(sd)
    mism = sorted(k for k in both if tuple(ours[k].shape) != tuple(sd[k].shape))
    return {"expected": len(ours), "matched": len(both) - len(mism), "missing": sorted(set(ours) - set(sd)),
            "unexpected": sorted(set(sd) - set(ours)), "shape_mismatch": mism}


def remap_keys(sd, rules):
    out = {}
    for k, v in sd.items():
        for old, new in rules:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


def safe_load(model, sd):
    r = load_report(model, sd)
    if r["missing"] or r["unexpected"] or r["shape_mismatch"]:
        raise RuntimeError(f"refusing to load: {r}")
    model.load_state_dict(sd, strict=True)
    return r["matched"]


@torch.no_grad()
def fraction_changed(before, model):
    after = model.state_dict()
    changed = sum(not torch.equal(before[k], after[k]) for k in before)
    return changed / len(before)
