# Pass 3 analysis: which heads track absolute length N smoothly?
# (a) first-token attention weight w0 vs N: a sink head has 1/w0 - 1 ~ N / e^D, i.e. log(1/w0 - 1) linear in log N, slope 1.
# (b) per-head output (o_proj input) late-vs-early mean difference, leave-one-trace-out separation.
import json, numpy as np
from wsp.editor import *
F = "work/fwd_03"; P = json.load(open("work/counter_points.json"))
off = np.load(f"{F}/offsets_tok.npy") if __import__("os").path.exists(f"{F}/offsets_tok.npy") else np.load(f"{F}/offsets.npy")
N = np.array([p["n_abs"] for p in P], float); NG = np.array([p["n_gen"] for p in P], float)
Q = np.array([p["id"] for p in P])
print("files", sorted(__import__("os").listdir(F)))
rows = []
for L in (2, 6, 10, 14, 18, 22):
    A = np.load(f"{F}/attn_last_L{L}.npy").astype(np.float64)
    for h in range(16):
        w0 = np.array([A[off[i], h] for i in range(len(P))])
        w0c = np.clip(w0, 1e-9, 1 - 1e-9); y = np.log(1 / w0c - 1); x = np.log(N)
        sl, ic = np.polyfit(x, y, 1); r = y - (sl * x + ic)
        rows.append((L, h, w0.mean(), sl, np.corrcoef(x, y)[0, 1], r.std()))
rows.sort(key=lambda r: -r[2])
print("heads by mean first-token weight (L, H, mean w0, slope of log(1/w0-1) on log N, corr, resid sd):")
for r in rows[:16]: print("  L%2d.H%2d  w0 %.3f  slope %.2f  corr %.2f  sd %.2f" % r)
# (b) head outputs
res = []
for L in (14, 18, 22):
    X = np.load(f"{F}/in_model.layers.{L}.self_attn.o_proj.npy").astype(np.float64).reshape(len(P), 16, 128)
    for h in range(16):
        Z = X[:, h]; S = np.zeros(len(P))
        for q in set(Q):
            tr = Q != q
            d = Z[tr & (NG >= 2300)].mean(0) - Z[tr & (NG <= 1300)].mean(0)
            S[Q == q] = (Z[Q == q] - Z[tr].mean(0)) @ d / (np.linalg.norm(d) + 1e-12)
        lo, hi = S[NG <= 1300], S[NG >= 2300]
        dp = (hi.mean() - lo.mean()) / np.sqrt(0.5 * (hi.var() + lo.var()) + 1e-12)
        res.append((dp, L, h, np.corrcoef(S, np.log(N))[0, 1]))
res.sort(reverse=True)
print("head outputs: held-out d' late (>=2300 gen) vs early (<=1300), corr with log N:")
for r in res[:10]: print("  d' %.2f  L%2d.H%2d  corr %.2f" % r)
