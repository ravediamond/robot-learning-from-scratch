"""PROVIDED toy pixel world. Do not edit.

A 2-D 'arm tip' moves toward a target. The target is visible ONLY in the image (a bright dot);
the state is the tip position. So a state-only policy cannot know where to go, and a policy
that uses pixels can. Image (3, S, S) in [0, 1]: channel 0 = target dot, channel 1 = tip dot.
Action chunk: the next H tip positions.
"""
import torch


def render(target: torch.Tensor, tip: torch.Tensor, S: int) -> torch.Tensor:
    img = torch.zeros(3, S, S)
    for ch, p in ((0, target), (1, tip)):
        r, c = int((p[1] + 1) / 2 * (S - 1)), int((p[0] + 1) / 2 * (S - 1))
        img[ch, max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = 1.0
    return img


def make_dot_data(n_episodes: int, T: int = 20, H: int = 4, S: int = 32, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    imgs, states, chunks, masks, ep_ids = [], [], [], [], []
    for e in range(n_episodes):
        target = torch.rand(2, generator=g) * 1.6 - 0.8
        tip = torch.rand(2, generator=g) * 1.6 - 0.8
        traj = [tip]
        for _ in range(T + H):
            traj.append(traj[-1] + 0.2 * (target - traj[-1]))
        traj = torch.stack(traj)                                    # (T+H+1, 2)
        for t in range(T):
            imgs.append(render(target, traj[t], S))
            states.append(traj[t])
            chunks.append(traj[t + 1:t + 1 + H])
            masks.append(torch.ones(H, dtype=torch.bool))
            ep_ids.append(e)
    return {"img": torch.stack(imgs), "state": torch.stack(states), "actions": torch.stack(chunks),
            "mask": torch.stack(masks), "episode": torch.tensor(ep_ids)}
