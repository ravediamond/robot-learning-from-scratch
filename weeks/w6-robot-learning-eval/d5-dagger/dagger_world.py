"""PROVIDED disturbance world for DAgger (uses sim2d). Do not edit.

Same obstacle world. The expert goes around the obstacle on the side of its current y (starts are
spread a few mm around y = 0, so demos cover both paths). During evaluation, at step KICK_T someone
bumps the arm DOWN by 0.3-0.5: the robot lands between the two demonstrated paths, close to the
obstacle, in states no demo visited. The expert knows what to do from there (go below);
behaviour cloning has never seen those states.
"""
import torch

import sim2d

KICK_T = 8


def expert(pos: torch.Tensor) -> torch.Tensor:
    """Raw expert action (N, 2): around the obstacle on the side of the current y."""
    return sim2d.expert_action(pos, torch.where(pos[:, 1] >= 0, 1.0, -1.0))


def expert_demos(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Undisturbed expert episodes from the usual starts: obs (N, 2), raw actions (N, 2)."""
    g = torch.Generator().manual_seed(seed)
    pos = sim2d.sample_starts(n, g)
    O, A = [], []
    for _ in range(sim2d.T_MAX):
        a = expert(pos)
        live = ~sim2d.reached(pos)
        O.append(pos[live])
        A.append(a[live])
        pos = pos + a
    return torch.cat(O), torch.cat(A)


def pushes(n: int, generator: torch.Generator) -> torch.Tensor:
    """(n, 2) downward bumps applied at step KICK_T."""
    return torch.stack([0.05 * torch.randn(n, generator=generator), -(0.3 + 0.2 * torch.rand(n, generator=generator))], -1)
