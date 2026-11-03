"""Run AFTER writing your guesses AND after your tests pass (~1 min)."""
import policies_ref as P
import sim2d
import w6d4_eval as m

demos = sim2d.make_demos(240, 8, seed=0)
val_ep = demos["episode"] >= 200
train = {k: v[~val_ep] for k, v in demos.items()}
val = {k: v[val_ep] for k, v in demos.items()}
mse, flow = P.train_mse_policy(train, 8), P.train_flow_policy(train, 8)
starts = m.fixed_starts(200, seed=1)
res = {}
for name, pol in [("MSE head", mse), ("flow head", flow)]:
    out = m.evaluate(pol.act, starts, 4, seed=2)
    k = int(out["success"].sum())
    res[name] = out
    print(f"{name:10s} val chunk MSE {m.val_chunk_mse(pol.act, val, seed=3):.2e}   success {k}/200 {tuple(round(x, 2) for x in sim2d.wilson(k, 200))}")
print("paired (A = MSE, B = flow):", m.paired_counts(res["MSE head"]["success"], res["flow head"]["success"]))
