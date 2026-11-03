"""W8 D5: checkpoint forensics."""
import torch
import torch.nn as nn


def load_report(model: nn.Module, sd: dict[str, torch.Tensor]) -> dict:
    """Compare a state dict with model.state_dict() WITHOUT loading.
    {"expected": n keys in the model, "matched": keys present in both with equal shapes,
     "missing": sorted model keys not in sd, "unexpected": sorted sd keys not in the model,
     "shape_mismatch": sorted keys in both with different shapes}  (matched is an int, the others lists)."""
    # TODO
    raise NotImplementedError


def remap_keys(sd: dict, rules: list[tuple[str, str]]) -> dict:
    """For each key, apply the FIRST rule (old_prefix, new_prefix) whose old_prefix starts the key
    (old_prefix may be ''). Keys with no matching rule are kept. Returns a new dict."""
    # TODO
    raise NotImplementedError


def safe_load(model: nn.Module, sd: dict) -> int:
    """Raise RuntimeError with the load_report if anything is missing, unexpected or shape-mismatched;
    otherwise load strictly and return the number of tensors loaded."""
    # TODO
    raise NotImplementedError


@torch.no_grad()
def fraction_changed(before: dict[str, torch.Tensor], model: nn.Module) -> float:
    """Fraction of tensors in `before` (a state_dict copy taken BEFORE loading) whose values changed."""
    # TODO
    raise NotImplementedError
