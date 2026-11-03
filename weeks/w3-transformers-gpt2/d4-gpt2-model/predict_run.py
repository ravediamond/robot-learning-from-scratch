"""Run AFTER writing your guesses in predict.md."""
import math

V, D, L, T = 50257, 768, 12, 1024
emb = V * D
block = 2 * 2 * D + (D * 3 * D + 3 * D) + (D * D + D) + (D * 4 * D + 4 * D) + (4 * D * D + D)
total = emb + T * D + L * block + 2 * D
print(f"1 wte {emb:,} = {emb / total:.0%}   2 per block {block:,}   total {total:,}")
print(f"3 ln(V) = {math.log(V):.2f}")
print(f"4 fp32 weights {total * 4 / 1e6:.0f} MB, Adam m+v +{2 * total * 4 / 1e6:.0f} MB (+ grads {total * 4 / 1e6:.0f} MB)")
print(f"5 untied {total + emb:,}")
