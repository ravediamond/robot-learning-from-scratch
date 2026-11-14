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
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = mlp(5, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    G = episode_returns(data, gamma)
    x = torch.cat([data["obs"], data["t"].float()[:, None] / 20], dim=1)
    for _ in range(steps):
        idx = torch.randint(0, len(G), (batch,), generator=g)
        loss = (net(x[idx]).squeeze(-1) - G[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def V(obs, t):
        with torch.no_grad():
            return net(torch.cat([obs, t.float()[:, None] / 20], dim=1)).squeeze(-1)

    return V


def n_step_advantage(data: dict, V, n: int = 5, gamma: float = 0.95) -> torch.Tensor:
    """A_t = sum_{k<m} gamma^k r_{t+k} + gamma^m V(s_{t+m}) - V(s_t), with m = min(n, T - t) and
    V = 0 beyond the last step (index T). The dataset is time-major when viewed as (T, n_episodes).
    Return (M,) in the dataset's flat order."""
    T = int(data["t"].max().item()) + 1
    n_ep = len(data["rew"]) // T
    r = data["rew"].view(T, n_ep)
    v = V(data["obs"], data["t"]).view(T, n_ep)
    v_pad = torch.cat([v, torch.zeros(1, n_ep)])  # V at index T is 0
    adv = torch.zeros(T, n_ep)
    for t in range(T):
        m = min(n, T - t)
        acc = torch.zeros(n_ep)
        for k in range(m):
            acc = acc + gamma ** k * r[t + k]
        adv[t] = acc + gamma ** m * v_pad[t + m] - v[t]
    return adv.flatten()


def improvement_indicator(adv: torch.Tensor, task_id: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    """I = adv > (the q-quantile of adv WITHIN the same task). Returns a bool tensor (M,)."""
    ind = torch.zeros_like(adv, dtype=torch.bool)
    for task in task_id.unique():
        mask = task_id == task
        ind[mask] = adv[mask] > torch.quantile(adv[mask], q)
    return ind


def train_conditioned(data: dict, indicator: torch.Tensor, steps: int = 2000, batch: int = 256, lr: float = 1e-3,
                      p_drop: float = 0.3, seed: int = 0):
    """Policy mlp(4 + 3, 2): input = obs + one-hot of the condition in {0: I=0, 1: I=1, 2: dropped}.
    Each batch: with probability p_drop (per sample) replace the condition by 2. MSE on the dataset action.
    Return act_fn(obs, i) with i in {0, 1, 2} (an int applied to the whole batch), no grad."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    pi = mlp(7, 2)
    opt = torch.optim.Adam(pi.parameters(), lr=lr)
    cond_all = indicator.long()
    n = len(cond_all)
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        cond = cond_all[idx].clone()
        cond[torch.rand(batch, generator=g) < p_drop] = 2
        x = torch.cat([data["obs"][idx], nn.functional.one_hot(cond, 3).float()], dim=1)
        loss = (pi(x) - data["act"][idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def act(obs, i: int):
        with torch.no_grad():
            c = nn.functional.one_hot(torch.full((len(obs),), i), 3).float()
            return pi(torch.cat([obs, c], dim=1))

    return act
