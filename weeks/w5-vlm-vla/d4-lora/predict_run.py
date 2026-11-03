"""Run AFTER writing your guesses in predict.md."""
print("1", 8 * (768 + 768), "vs", 768 * 768)
print("2", 12 * 2 * 16 * (768 + 768))
print(f"3 full: weights grads m v = 4 x 450M x 4 B = {4 * 450e6 * 4 / 1e9:.1f} GB;"
      f" LoRA: 450M x 4 B (frozen weights) + 3 x 10M x 4 B = {(450e6 * 4 + 3 * 10e6 * 4) / 1e9:.2f} GB (+ activations)")
print("4 equal: B = 0 at init")
