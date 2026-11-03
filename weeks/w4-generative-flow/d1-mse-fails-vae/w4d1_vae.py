"""W4 D1: SOLUTION."""
import torch
import torch.nn as nn


def mse_optimum(y):
    return y.mean(dim=0)


def reparameterize(mu, logvar, eps):
    return mu + torch.exp(0.5 * logvar) * eps


def kl_to_standard_normal(mu, logvar):
    return 0.5 * (logvar.exp() + mu ** 2 - 1.0 - logvar).sum(-1)


class VAE(nn.Module):
    def __init__(self, D=2, Z=2, hidden=64):
        super().__init__()
        self.Z = Z
        self.enc = nn.Sequential(nn.Linear(D, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2 * Z))
        self.dec = nn.Sequential(nn.Linear(Z, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, D))

    def encode(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z):
        return self.dec(z)

    def forward(self, x, eps):
        mu, logvar = self.encode(x)
        return self.decode(reparameterize(mu, logvar, eps)), mu, logvar

    @torch.no_grad()
    def sample(self, n, generator=None):
        return self.decode(torch.randn(n, self.Z, generator=generator))


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = ((x_hat - x) ** 2).sum(-1)
    return (recon + beta * kl_to_standard_normal(mu, logvar)).mean()


def train_vae(x, steps=3000, beta=0.05, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    vae = VAE(D=x.shape[1])
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        eps = torch.randn(len(x), vae.Z, generator=g)
        x_hat, mu, logvar = vae(x, eps)
        loss = vae_loss(x, x_hat, mu, logvar, beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vae.eval()
