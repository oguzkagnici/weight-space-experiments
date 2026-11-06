# Combine signals: (a) summed sink log-odds over L18 H0/H3/H5/H6/H7 and L2.H2; (b) L18 all-head output; (c) residuals.
# Held-out (leave-one-trace-out) separation between early (<=1300 generated tokens) and late (>=2300).
import json, numpy as np
F = "work/fwd_03"; P = json.load(open("work/counter_points.json")); off = np.load(f"{F}/offsets_tok.npy")
N = np.array([p["n_abs"] for p in P], float); NG = np.array([p["n_gen"] for p in P], float); Q = np.array([p["id"] for p in P])
def sep(S, name):
    lo, hi = S[NG <= 1300], S[NG >= 2300]
    dp = (hi.mean() - lo.mean()) / np.sqrt(0.5 * (hi.var() + lo.var()))
    ov = (hi.min() > lo.max())
    by = {t: S[np.array([p["target"] for p in P]) == t].round(2).tolist() for t in (600, 1200, 1800, 2400, 2900)}
    print(f"{name:28s} d' {dp:5.2f}  min(late) {hi.min():6.2f}  max(early) {lo.max():6.2f}  clean split: {ov}")
    return by
sink = 0
for L, h in [(18, 0), (18, 3), (18, 5), (18, 6), (18, 7), (2, 2)]:
    A = np.load(f"{F}/attn_last_L{L}.npy").astype(np.float64)
    w0 = np.clip(np.array([A[off[i], h] for i in range(len(P))]), 1e-9, 1)
    sink = sink + np.log(1 / w0 - 1)
sep(sink, "summed sink log-odds")
sep(np.log(N), "log N (oracle)")
def loo(Z, name):
    S = np.zeros(len(P))
    for q in set(Q):
        tr = Q != q; d = Z[tr & (NG >= 2300)].mean(0) - Z[tr & (NG <= 1300)].mean(0)
        S[Q == q] = (Z[Q == q] - Z[tr].mean(0)) @ d / np.linalg.norm(d)
    return sep(S, name)
for L in (18, 22):
    loo(np.load(f"{F}/in_model.layers.{L}.self_attn.o_proj.npy").astype(np.float64), f"L{L} all-head output")
for L in (18, 20, 22):
    H = np.load(f"{F}/h{L}.npy").astype(np.float64); H /= np.sqrt((H ** 2).mean(1, keepdims=True))
    by = loo(H, f"residual h{L} (normed)")
print("h22 held-out scores by target length:", by)
