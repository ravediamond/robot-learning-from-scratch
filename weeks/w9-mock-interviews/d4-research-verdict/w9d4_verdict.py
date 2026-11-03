"""W9 D4: the three verdicts as code."""
import torch


def paired_diff_ci(base: torch.Tensor, method: torch.Tensor, B: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(d = mean(method - base), lo, hi): 95% percentile bootstrap over the paired starts."""
    # TODO
    raise NotImplementedError


def verdict(d: float, lo: float, hi: float, pass_line: float, fail_line: float, confounds: list[str] | None = None) -> str:
    """'pass', 'fail' or 'not proven' (README rule)."""
    # TODO
    raise NotImplementedError
