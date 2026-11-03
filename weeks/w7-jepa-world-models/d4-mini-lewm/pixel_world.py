"""PROVIDED pixel world for the JEPA world model (W7 D4-D5). Do not edit.

A point moves in the box [-0.9, 0.9]^2. Action = displacement (2,), each component in [-0.1, 0.1].
Observation = a 32x32 grey image with a soft blob (Gaussian, sigma 1.5 px) at the point.
The model never sees the position: only pixels and actions. Position is used only to TEST it (probes,
planning success).
"""
import torch

S, LIM, AMAX = 32, 0.9, 0.1
_g = torch.linspace(-1, 1, S)
_YY, _XX = torch.meshgrid(_g, _g, indexing="ij")


def render(pos: torch.Tensor) -> torch.Tensor:
    """(N, 2) -> (N, 1, S, S) in [0, 1]."""
    sig = 1.5 * 2 / (S - 1)
    d2 = (_XX[None] - pos[:, 0, None, None]) ** 2 + (_YY[None] - pos[:, 1, None, None]) ** 2
    return torch.exp(-d2 / (2 * sig ** 2))[:, None]


def step(pos: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return (pos + action.clamp(-AMAX, AMAX)).clamp(-LIM, LIM)


def make_episodes(n_episodes: int, T: int, seed: int = 0) -> dict:
    """Random-action episodes. Returns pos (E, T+1, 2), action (E, T, 2), obs (E, T+1, 1, S, S)."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(n_episodes, 2, generator=g) * 1.6 - 0.8
    P, A = [pos], []
    for _ in range(T):
        a = (torch.rand(n_episodes, 2, generator=g) * 2 - 1) * AMAX
        pos = step(pos, a)
        P.append(pos)
        A.append(a)
    P, A = torch.stack(P, 1), torch.stack(A, 1)
    return {"pos": P, "action": A, "obs": render(P.reshape(-1, 2)).view(n_episodes, T + 1, 1, S, S)}
