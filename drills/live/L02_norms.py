"""L02 (30 min). LayerNorm and RMSNorm over the last dimension, from scratch (no F.layer_norm)."""
import torch
from torch import nn


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * weight + bias, biased variance (unbiased=False). weight = 1, bias = 0 at init."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * weight. No centring, no bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
