"""Run AFTER writing your guesses AND after your tests pass (~2 min)."""
import pixel_world as W
import w7d4_lewm as m

tr, va = W.make_episodes(200, 20, seed=0), W.make_episodes(40, 20, seed=1)
for lam, steps in [(0.0, 500), (0.1, 1500), (1.0, 1500)]:
    enc, pred = m.train_lewm(tr, lam, steps=steps)
    r = m.latent_report(enc, pred, tr, va)
    print(f"lambda {lam}: " + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
