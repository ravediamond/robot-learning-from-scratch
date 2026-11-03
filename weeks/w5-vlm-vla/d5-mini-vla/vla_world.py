"""PROVIDED two-target world for the mini VLA. Do not edit.

A point robot and two targets (red, blue) at random places. The instruction says which target to go to:
tokens ["go", "to", "the", "red"|"blue"]. Only the IMAGE shows where the targets are, only the
INSTRUCTION says which one: a policy needs both.

Image (3, 16, 16): channel 0 red target, 1 blue target, 2 robot (3x3 squares).
Action: velocity (2,), clipped to STEP. Success: within TOL of the instructed target.
"""
import torch

VOCAB = ["go", "to", "the", "red", "blue", "<pad>"]
S, STEP, TOL, T_MAX = 16, 0.08, 0.1, 30


def tokens_for(color: torch.Tensor) -> torch.Tensor:
    """color (N,) 0 = red, 1 = blue -> (N, 4) token ids."""
    base = torch.tensor([0, 1, 2, 3]).repeat(len(color), 1)
    base[:, 3] = 3 + color
    return base


def render(red: torch.Tensor, blue: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
    """(N, 2) each, coordinates in [-1, 1] -> (N, 3, S, S)."""
    N = len(red)
    img = torch.zeros(N, 3, S, S)
    for ch, p in enumerate((red, blue, robot)):
        rc = ((p + 1) / 2 * (S - 1)).round().long().clamp(1, S - 2)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                img[torch.arange(N), ch, rc[:, 1] + dr, rc[:, 0] + dc] = 1.0
    return img


def clip(a: torch.Tensor) -> torch.Tensor:
    n = a.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return a * (n.clamp(max=STEP) / n)


def sample_scenes(n: int, g: torch.Generator):
    red = torch.rand(n, 2, generator=g) * 1.6 - 0.8
    blue = torch.rand(n, 2, generator=g) * 1.6 - 0.8
    close = (red - blue).norm(dim=-1) < 0.6
    blue[close] = -red[close]                                        # keep the targets apart
    blue = blue.clamp(-0.8, 0.8)
    robot = torch.rand(n, 2, generator=g) * 1.6 - 0.8
    color = torch.randint(0, 2, (n,), generator=g)
    return red, blue, robot, color


def make_vla_demos(n_episodes: int, H: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    red, blue, pos, color = sample_scenes(n_episodes, g)
    target = torch.where(color[:, None] == 0, red, blue)
    out = {k: [] for k in ("img", "tokens", "state", "chunk")}
    traj = [pos]
    for _ in range(T_MAX + H):
        traj.append(traj[-1] + clip(target - traj[-1]))
    traj = torch.stack(traj, 1)                                       # (E, T+H+1, 2)
    acts = traj[:, 1:] - traj[:, :-1]                                 # (E, T+H, 2)
    for t in range(T_MAX):
        live = (traj[:, t] - target).norm(dim=-1) > TOL / 2
        if not live.any():
            break
        out["img"].append(render(red[live], blue[live], traj[live, t]))
        out["tokens"].append(tokens_for(color[live]))
        out["state"].append(traj[live, t])
        out["chunk"].append(acts[live, t:t + H] / STEP)               # training units: [-1, 1]
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def run_episodes(act_fn, n: int, seed: int = 0, swap_instruction: bool = False, H_exec: int = 2) -> dict:
    """act_fn(img, tokens, state, generator) -> chunk in training units (N, H, 2).
    Returns {"success": reached the instructed target, "other": reached the other target} (n,) bool.
    swap_instruction=True asks for the OTHER colour (success is then measured against the asked one)."""
    g = torch.Generator().manual_seed(seed)
    red, blue, pos, color = sample_scenes(n, g)
    if swap_instruction:
        color = 1 - color
    target = torch.where(color[:, None] == 0, red, blue)
    other = torch.where(color[:, None] == 0, blue, red)
    succ, oth = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    t = 0
    while t < T_MAX:
        live = ~(succ | oth)
        if not live.any():
            break
        chunk = act_fn(render(red[live], blue[live], pos[live]), tokens_for(color[live]), pos[live], g)
        idx = live.nonzero().squeeze(1)
        for k in range(min(H_exec, chunk.shape[1])):
            still = ~(succ[idx] | oth[idx])
            pos[idx] = pos[idx] + clip(chunk[:, k] * STEP) * still[:, None]
            succ[idx] |= ((pos[idx] - target[idx]).norm(dim=-1) < TOL) & still
            oth[idx] |= ((pos[idx] - other[idx]).norm(dim=-1) < TOL) & still & ~succ[idx]
            t += 1
    return {"success": succ, "other": oth}
