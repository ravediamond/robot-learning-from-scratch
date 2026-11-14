"""Provided: a mixed-quality offline dataset on Reach2D."""
import torch

import rl_envs


def make_dataset(n_episodes: int = 600, horizon: int = 20, seed: int = 0, per_step: bool = False) -> dict:
    """25% expert, 25% random, 50% anti-expert (walks away from the goal), all with small action noise.
    Returns flat tensors: obs (M, 4), act (M, 2), rew (M,), next_obs (M, 4), done (M,) (always False:
    time-limit only), plus episode (M,) and t (M,) indices and kind (M,) in {0 expert, 1 random, 2 anti}.
    per_step=False: one source per EPISODE (W10 D4). per_step=True: the source is re-drawn at every STEP with
    the same proportions (W10 D5), so the state no longer tells you who is acting."""
    g = torch.Generator().manual_seed(seed)
    env = rl_envs.Reach2D(n_episodes, horizon, seed=seed)
    kind = torch.zeros(n_episodes, dtype=torch.long)
    kind[n_episodes // 4: n_episodes // 2] = 1
    kind[n_episodes // 2:] = 2
    obs = env.reset()
    rows = {k: [] for k in ("obs", "act", "rew", "next_obs", "t", "kind")}
    for t in range(horizon):
        if per_step:
            u = torch.rand(n_episodes, generator=g)
            kind = (u > 0.25).long() + (u > 0.5).long()
        e = rl_envs.expert_action(obs)
        a = torch.where(kind[:, None] == 0, e, torch.where(kind[:, None] == 1, torch.rand(n_episodes, 2, generator=g) * 2 - 1, -e))
        a = (a + 0.1 * torch.randn(n_episodes, 2, generator=g)).clamp(-1, 1)
        nxt, r, _ = env.step(a)
        for k, v in (("obs", obs), ("act", a), ("rew", r), ("next_obs", nxt), ("t", torch.full((n_episodes,), t)),
                     ("kind", kind.clone())):
            rows[k].append(v)
        obs = nxt
    out = {k: torch.cat(v) for k, v in rows.items()}
    out["done"] = torch.zeros(len(out["rew"]), dtype=torch.bool)
    out["episode"] = torch.arange(n_episodes).repeat(horizon)
    return out
