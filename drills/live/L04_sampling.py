"""L04 (30 min). Top-k and top-p (nucleus) filtering of logits, then sampling."""
import torch


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """logits (B, V). Return a copy where removed entries are -inf.
    top_k > 0: keep the k largest per row (ties at the boundary: keep all tied values).
    top_p < 1: after top-k, sort descending, keep the smallest prefix whose softmax probabilities sum to >= top_p
    (always keep at least the top token)."""
    out = logits.clone()
    if top_k > 0:
        kth = torch.topk(out, top_k, dim=-1).values[:, -1:]
        out = out.masked_fill(out < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(out, dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        remove = cum - probs >= top_p            # tokens AFTER the prefix already reached top_p
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        out = torch.full_like(out, float("-inf")).scatter(1, idx, sorted_logits)
    return out


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
           generator: torch.Generator) -> torch.Tensor:
    """Divide by temperature (temperature 0 means greedy argmax), filter, sample one token per row with
    torch.multinomial(probs, 1, generator=generator). Return (B,) token ids."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = filter_logits(logits / temperature, top_k, top_p).softmax(dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)
