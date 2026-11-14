"""W10 D5: advantage-conditioned imitation (RECAP-style) on the per-step mixed dataset."""
import torch
from torch import nn


def mlp(inp: int, out: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


def episode_returns(data: dict, gamma: float) -> torch.Tensor:
    """Provided: discounted return-to-go G (M,) for the flat dataset (episodes all have the same length)."""
    T = int(data["t"].max().item()) + 1
    n_ep = len(data["rew"]) // T
    r = data["rew"].view(T, n_ep)
    G = torch.zeros_like(r)
    run = torch.zeros(n_ep)
    for t in range(T - 1, -1, -1):
        run = r[t] + gamma * run
        G[t] = run
    return G.flatten()


def fit_value(data: dict, gamma: float = 0.95, steps: int = 1500, batch: int = 256, lr: float = 1e-3, seed: int = 0):
    """Regress an mlp(5, 1) on (obs, t / 20) -> return-to-go (MSE). Time is an input because the horizon is
    fixed. Return V(obs (B, 4), t (B,)) -> (B,) values, no grad."""
    # TODO
    raise NotImplementedError


def n_step_advantage(data: dict, V, n: int = 5, gamma: float = 0.95) -> torch.Tensor:
    """A_t = sum_{k<m} gamma^k r_{t+k} + gamma^m V(s_{t+m}) - V(s_t), with m = min(n, T - t) and
    V = 0 beyond the last step (index T). The dataset is time-major when viewed as (T, n_episodes).
    Return (M,) in the dataset's flat order."""
    # TODO
    raise NotImplementedError


def improvement_indicator(adv: torch.Tensor, task_id: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    """I = adv > (the q-quantile of adv WITHIN the same task). Returns a bool tensor (M,)."""
    # TODO
    raise NotImplementedError


def train_conditioned(data: dict, indicator: torch.Tensor, steps: int = 2000, batch: int = 256, lr: float = 1e-3,
                      p_drop: float = 0.3, seed: int = 0):
    """Policy mlp(4 + 3, 2): input = obs + one-hot of the condition in {0: I=0, 1: I=1, 2: dropped}.
    Each batch: with probability p_drop (per sample) replace the condition by 2. MSE on the dataset action.
    Return act_fn(obs, i) with i in {0, 1, 2} (an int applied to the whole batch), no grad."""
    # TODO
    raise NotImplementedError
