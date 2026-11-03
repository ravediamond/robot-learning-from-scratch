import torch


@torch.no_grad()
def cem(dynamics, cost, z0, H, A, N, K, iters, a_max, generator):
    mu, std = torch.zeros(H, A), torch.full((H, A), a_max)
    for _ in range(iters):
        a = (mu + std * torch.randn(N, H, A, generator=generator)).clamp(-a_max, a_max)
        z = z0.expand(N, -1)
        c = torch.zeros(N)
        for h in range(H):
            z = dynamics(z, a[:, h])
            c = c + cost(z)
        el = a[c.topk(K, largest=False).indices]
        mu, std = el.mean(0), el.std(0) + 1e-4
    return mu
