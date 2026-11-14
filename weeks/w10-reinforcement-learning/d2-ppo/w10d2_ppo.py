"""W10 D2: PPO on Reach2D."""
import torch

import rl_envs
from w10d2_helpers import GaussianPolicy, collect, gae, value_net


def ppo_clip_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, adv: torch.Tensor, clip: float = 0.2) -> torch.Tensor:
    """-mean(min(ratio * adv, clamp(ratio, 1 - clip, 1 + clip) * adv)), ratio = exp(logp_new - logp_old)."""
    # TODO
    raise NotImplementedError


def normalize(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero mean, unit std (over all elements)."""
    # TODO
    raise NotImplementedError


def value_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """0.5 * mean squared error."""
    # TODO
    raise NotImplementedError


def ppo_update(policy, value, opt, batch: dict, gamma: float = 0.99, lam: float = 0.95, clip: float = 0.2,
               epochs: int = 4, minibatches: int = 4, vf_coef: float = 0.5, generator=None) -> None:
    """One PPO update from a collected batch (see w10d2_helpers.collect).
    1. adv = gae(...) on the stored values; v_target = adv + stored values. Flatten (T, N, ...) to (T*N, ...).
    2. For each epoch: shuffle indices (torch.randperm with `generator`), split into `minibatches` chunks.
    3. Per chunk: loss = ppo_clip_loss(new logp, stored logp, normalize(adv chunk), clip)
                        + vf_coef * value_loss(value(obs), v_target chunk); step `opt`."""
    # TODO
    raise NotImplementedError


def train_ppo(iters: int = 40, n_envs: int = 256, lr: float = 3e-3, seed: int = 0):
    """Build policy + value net, one Adam over both, then `iters` x (collect, ppo_update).
    Return (policy, history of mean episode returns of each collected batch)."""
    # TODO
    raise NotImplementedError
