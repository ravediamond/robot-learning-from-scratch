"""W10 D2: PPO on Reach2D."""
import torch

import rl_envs
from w10d2_helpers import GaussianPolicy, collect, gae, value_net


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> torch.Tensor:
    """-mean(min(ratio * adv, clamp(ratio, 1 - clip, 1 + clip) * adv)), ratio = exp(logp_new - logp_old)."""
    ratio = (logp_new - logp_old).exp()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip, 1 + clip) * adv
    return -torch.min(surr1, surr2).mean()


def normalize(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero mean, unit std (over all elements)."""
    return (adv - adv.mean()) / (adv.std() + eps)


def value_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """0.5 * mean squared error."""
    return 0.5 * (v_pred - v_target).pow(2).mean()


def ppo_update(policy, value, opt, batch: dict, gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2,
               epochs: int = 4, minibatches: int = 4, vf_coef: float = 0.5, generator=None) -> None:
    """One PPO update from a collected batch (see w10d2_helpers.collect).
    1. adv = gae(...) on the stored values; v_target = adv + stored values. Flatten (T, N, ...) to (T*N, ...).
    2. For each epoch: shuffle indices (torch.randperm with `generator`), split into `minibatches` chunks.
    3. Per chunk: loss = ppo_clip_loss(new logp, stored logp, normalize(adv chunk), clip)
                        + vf_coef * value_loss(value(obs), v_target chunk); step `opt`."""
    adv = gae(batch["rew"], batch["val"], batch["done"], batch["last_val"], gamma, lam)  # (T, N)
    v_target = adv + batch["val"]
    obs = batch["obs"].flatten(0, 1)        # (T*N, D)
    act = batch["act"].flatten(0, 1)        # (T*N, A)
    logp_old = batch["logp"].flatten()      # (T*N,)
    adv, v_target = adv.flatten(), v_target.flatten()
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=generator)
        for idx in perm.chunk(minibatches):
            loss = ppo_clip_loss(policy.log_prob(obs[idx], act[idx]), logp_old[idx], normalize(adv[idx]), clip)
            loss = loss + vf_coef * value_loss(value(obs[idx]).squeeze(-1), v_target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


def train_ppo(iters: int = 40, n_envs: int = 256, lr: float = 3e-3, seed: int = 0):
    """Build policy + value net, one Adam over both, then `iters` x (collect, ppo_update).
    Return (policy, history of mean episode returns of each collected batch)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    policy = GaussianPolicy(rl_envs.Reach2D.obs_dim, rl_envs.Reach2D.act_dim)
    value = value_net(rl_envs.Reach2D.obs_dim)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr)
    env = rl_envs.Reach2D(n_envs, seed=seed)
    history = []
    for _ in range(iters):
        batch = collect(env, policy, value)
        history.append(batch["rew"].sum(dim=0).mean().item())
        ppo_update(policy, value, opt, batch, generator=g)
    return policy, history
