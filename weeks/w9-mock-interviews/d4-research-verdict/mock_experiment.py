"""PROVIDED mock experiment. A paper claims its method adds +15 points of success. You ran a paired test
on 30 fixed starts. Do not read the generator to "check" the truth: decide with your rule.
Pass line P = 0.10, fail line F = 0.02 (written in the protocol before the run)."""
import torch

PASS_LINE, FAIL_LINE = 0.10, 0.02


def mock_experiment(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    difficulty = torch.rand(30, generator=g)
    base = (torch.rand(30, generator=g) > difficulty * 0.9).float()
    method = (torch.rand(30, generator=g) > difficulty * 0.75).float()
    return base, method
