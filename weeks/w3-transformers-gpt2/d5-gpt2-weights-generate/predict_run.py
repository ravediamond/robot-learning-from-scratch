"""Run AFTER writing your guesses in predict.md."""
import math
from collections import Counter
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel

hf = GPT2LMHeadModel(GPT2Config(n_layer=1))
print("1 HF c_attn:", tuple(hf.state_dict()["transformer.h.0.attn.c_attn.weight"].shape), " yours: (2304, 768)")
print("2 a square weight loads without error even untransposed: only the logits check catches it")
print("3 greedy: same text; sampling: different text (unless you fix the generator seed)")
book = Path(__file__).resolve().parents[3] / "reference" / "journal" / "book"
text = "".join(p.read_text() for p in sorted(book.glob("*.md")))
c = Counter(text)
H = -sum(n / len(text) * math.log(n / len(text)) for n in c.values())
print(f"4 book: {len(text):,} chars, vocab {len(c)}, unigram entropy {H:.2f} nats")
print("5 without cache:", sum(20 + i for i in range(200)), "positions; with a KV cache: 20 + 200 =", 220)
