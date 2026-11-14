"""W14 D2: symmetric int8 quantisation, calibration, fake quantisation with a straight-through estimator."""
import torch
from torch import nn

QMAX = 127


def quantize(w: torch.Tensor, per_channel: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8. scale = max|w| / 127, per output channel (dim 0, shape (C, 1, ..., 1)) or one scalar.
    q = clamp(round(w / scale), -127, 127) as torch.int8. A zero channel uses scale 1.0. Return (q, scale)."""
    if per_channel:
        amax = w.abs().reshape(w.shape[0], -1).max(dim=1).values.reshape(-1, *([1] * (w.dim() - 1)))
    else:
        amax = w.abs().max()
    scale = torch.where(amax > 0, amax / QMAX, torch.ones_like(amax))
    q = torch.clamp(torch.round(w / scale), -QMAX, QMAX).to(torch.int8)
    return q, scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32) * scale


def percentile_scale(x: torch.Tensor, pct: float) -> float:
    """Scale from the pct-th percentile of |x| (torch.quantile on a float tensor, q = pct / 100), divided by 127."""
    return torch.quantile(x.abs().float(), pct / 100).item() / QMAX


def fake_quant(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantise-dequantise in the forward pass (clamp to [-127, 127] after rounding), identity gradient
    (straight-through estimator) inside the clamp range: return x + (xq - x).detach()."""
    xq = torch.clamp(torch.round(x / scale), -QMAX, QMAX) * scale
    return x + (xq - x).detach()


@torch.no_grad()
def quantize_linear_weights(model: nn.Module, per_channel: bool = True) -> dict[str, float]:
    """Replace every nn.Linear weight IN PLACE by its int8 round trip (dequantize(quantize(w))).
    Return {module name: relative error mean|w_q - w| / mean|w|} for each Linear."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight
            wq = dequantize(*quantize(w, per_channel))
            out[name] = ((wq - w).abs().mean() / w.abs().mean()).item()
            w.copy_(wq)
    return out
