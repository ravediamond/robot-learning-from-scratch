"""PROVIDED driver: train YOUR GPT2 on your own book, char-level, then sample.
    uv run python train_char.py [--steps 3000]
~10 min on Apple MPS. Watch val loss go below the unigram entropy, then below ~1.5 nats."""
import argparse
import math
import time
from pathlib import Path

import torch

import w3d5_load_generate as m
from w3d4_gpt2 import GPT2, GPT2Config

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
a = ap.parse_args()
dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
book = Path(__file__).resolve().parents[3] / "reference" / "journal" / "book"
text = "".join(p.read_text() for p in sorted(book.glob("*.md")))
tok = m.CharTokenizer(text)
data = torch.tensor(tok.encode(text))
n = int(0.9 * len(data))
train, val = data[:n], data[n:]
cfg = GPT2Config(vocab_size=tok.vocab_size, n_positions=128, n_embd=192, n_layer=4, n_head=6)
model = GPT2(cfg).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
g = torch.Generator().manual_seed(0)
t0 = time.time()
for step in range(a.steps + 1):
    lr = 1e-3 * min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
    for grp in opt.param_groups:
        grp["lr"] = lr
    x, y = m.get_batch(train, 128, 32, g)
    loss = model(x.to(dev), y.to(dev))[1]
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = m.get_batch(val, 128, 64, torch.Generator().manual_seed(1))
            vl = model(xv.to(dev), yv.to(dev))[1].item()
        model.train()
        print(f"step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  {time.time() - t0:.0f}s")
model.eval()
prompt = torch.tensor([tok.encode("The baseline ")], device=dev)
out = m.generate(model, prompt, 300, temperature=0.8, top_k=20, generator=torch.Generator(device=dev).manual_seed(0) if dev.type != "mps" else None)
print("\n" + tok.decode(out[0].tolist()))
