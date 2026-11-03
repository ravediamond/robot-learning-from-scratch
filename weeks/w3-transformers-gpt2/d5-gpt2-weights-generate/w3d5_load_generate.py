"""W3 D5: SOLUTION."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "d4-gpt2-model"))
from w3d4_gpt2 import GPT2, GPT2Config  # noqa: E402,F401

CONV1D = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")


@torch.no_grad()
def load_hf_state_dict(model, hf_sd):
    ours = model.state_dict()
    filled, n = set(), 0
    for k, v in hf_sd.items():
        if k.endswith(".attn.bias") or k.endswith(".attn.masked_bias") or k == "lm_head.weight":
            continue
        name = k.removeprefix("transformer.")
        if name not in ours:
            raise ValueError(f"unexpected key {k}")
        if name.endswith(CONV1D):
            v = v.t()
        if v.shape != ours[name].shape:
            raise ValueError(f"shape mismatch {name}: {tuple(v.shape)} vs {tuple(ours[name].shape)}")
        ours[name].copy_(v)
        filled.add(name)
        n += 1
    missing = set(ours) - filled - {"lm_head.weight"}
    if missing:
        raise ValueError(f"not filled: {sorted(missing)[:5]} ... ({len(missing)})")
    return n


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, generator=None):
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.cfg.n_positions:])[0][:, -1]           # (B, V)
        if temperature == 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def get_batch(data, block, batch, generator):
    i = torch.randint(0, len(data) - block - 1, (batch,), generator=generator)
    x = torch.stack([data[j:j + block] for j in i.tolist()])
    y = torch.stack([data[j + 1:j + block + 1] for j in i.tolist()])
    return x, y
