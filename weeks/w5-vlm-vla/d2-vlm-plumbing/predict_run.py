"""Run AFTER writing your guesses in predict.md."""
N = (512 // 16) ** 2
print(f"1 tokens {N} -> {N // 16}, channels {768 * 16}")
print("2 total", 2 * 64 + 20 + 1 + 50)
P, S, A = 148, 1, 50
blocks = P * P + S * (P + S) + A * (P + S + A)
T = P + S + A
print(f"3 block mask allows {blocks:,} pairs; causal allows {T * (T + 1) // 2:,}; full {T * T:,}")
print("4 no: prefix rows cannot see later blocks, so the prefix K/V can be cached across flow steps")
