"""Toy SO-101 demonstrations. PROVIDED: read it, do not edit it.

6 dims like the real arm: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
(degrees, about -150..150) and gripper (0 = closed .. 100 = open). Note the different scales:
that is why normalization matters.

Each episode: the "operator" moves from a random start to a random goal pose with a smooth,
slightly noisy proportional controller, then closes the gripper near the goal.
Observation = current state (6) + goal pose (6), like a state + instruction.
Action = next commanded joint positions (6), as in LeRobot (absolute targets, not deltas).
"""
import torch

JOINT_LIMITS = torch.tensor([[-150, 150], [-100, 100], [-100, 100], [-100, 100], [-150, 150], [0, 100]],
                            dtype=torch.float32)


def generate_episodes(n_episodes: int = 60, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    lo, hi = JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]
    episodes = []
    for _ in range(n_episodes):
        T = int(torch.randint(60, 121, (1,), generator=g))
        start = lo + (hi - lo) * (0.2 + 0.6 * torch.rand(6, generator=g))
        goal = lo + (hi - lo) * (0.2 + 0.6 * torch.rand(6, generator=g))
        start[5], goal[5] = 100.0, 100.0  # gripper open
        gain = 0.15 + 0.10 * torch.rand(1, generator=g)  # each operator a bit different
        s = start.clone()
        states, actions = [], []
        for t in range(T):
            target = s + gain * (goal - s)
            near = (goal[:5] - s[:5]).abs().max() < 8.0
            target[5] = s[5] - 10.0 if near else s[5]                  # close gripper near goal
            target = target + 0.5 * torch.randn(6, generator=g)        # operator noise
            target = torch.maximum(torch.minimum(target, hi), lo)
            states.append(s.clone())
            actions.append(target.clone())
            s = s + 0.5 * (target - s)                                  # servo tracks 50% per step (lag)
        st = torch.stack(states)                                         # (T, 6)
        obs = torch.cat([st, goal.expand(T, 6)], dim=1)                  # (T, 12)
        episodes.append({"obs": obs, "action": torch.stack(actions)})    # (T, 12), (T, 6)
    return episodes
