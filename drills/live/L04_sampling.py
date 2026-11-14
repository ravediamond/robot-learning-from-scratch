"""L04 (30 min). Top-k and top-p (nucleus) filtering of logits, then sampling."""
import torch


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """logits (B, V). Return a copy where removed entries are -inf.
    top_k > 0: keep the k largest per row (ties at the boundary: keep all tied values).
    top_p < 1: after top-k, sort descending, keep the smallest prefix whose softmax probabilities sum to >= top_p
    (always keep at least the top token)."""
    # TODO
    raise NotImplementedError


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
           generator: torch.Generator) -> torch.Tensor:
    """Divide by temperature (temperature 0 means greedy argmax), filter, sample one token per row with
    torch.multinomial(probs, 1, generator=generator). Return (B,) token ids."""
    # TODO
    raise NotImplementedError
