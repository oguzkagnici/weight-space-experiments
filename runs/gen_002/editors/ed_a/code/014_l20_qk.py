# Hypothesis: sliding-window L20 heads have the same primacy mechanism (constant q along an ordinal key direction). Fit positional encoding, slot-mean keys, q.v1.
import json, numpy as np
exec(open("code/008_rope_fit.py").read().split("variants =")[0].split("W = ")[0])  # imports only
W = "work/fwd_05/"
q = np.load(W+"out_model.layers.20.self_attn.q_norm.npy").astype(np.float32)
k = np.load(W+"out_model.layers.20.self_attn.k_norm.npy").astype(np.float32)
A = np.load(W+"attn_last_L20.npy").astype(np.float32); off = np.load(W+"offsets_tok.npy")
spans = json.load(open("work/p2_spans.json"))
src = open("code/008_rope_fit.py").read(); exec(src[src.index("def rope"):src.index("variants =")])
for name, th, fr, st in [("nope", 1, 0, "half"), ("half 1e4 f1", 1e4, 1.0, "half"), ("half 1e6 f1", 1e6, 1.0, "half"), ("inter 1e4 f1", 1e4, 1.0, "inter"), ("half 1e4 f.5", 1e4, .5, "half")]:
    errs = []
    for i in range(40):
        a, b = off[i], off[i+1]; pos = np.arange(b-a, dtype=np.float64)
        qq = rope(q[b-1:b], pos[-1:], th, fr, st)[0]; kk = np.repeat(rope(k[a:b], pos, th, fr, st), 4, axis=1)
        s = (kk * qq[None]).sum(-1) / np.sqrt(128); la = np.log(np.maximum(A[a:b], 1e-8)); m = A[a:b] > 1e-4
        for h in [2, 4, 12, 13]:
            x, y = s[m[:, h], h], la[m[:, h], h]; x, y = x-x.mean(), y-y.mean(); sl = (x@y)/(x@x)
            errs.append(np.mean((y-sl*x)**2)/np.var(y))
    print(f"{name:14s} resid frac {np.mean(errs):.3f}")
mu = np.zeros((120, 10, 4, 128))
for qi in range(120):
    for kk_, (t0, t1) in enumerate(spans[qi]): mu[qi, kk_] = k[off[qi]+t0:off[qi]+t1].mean(0)
M = mu.mean(0); Mc = M - M.mean(0); np.save("work/p5_slotkeys_L20.npy", M)
qlast = np.stack([q[off[i+1]-1] for i in range(120)])
np.set_printoptions(precision=2, suppress=True, linewidth=200)
M1, M2 = mu[:60].mean(0), mu[60:].mean(0); M1 -= M1.mean(0); M2 -= M2.mean(0)
for g in range(4):
    U, s, Vt = np.linalg.svd(Mc[:, g], full_matrices=False)
    print(f"kv{g}: sv {s[:4]} split-half r {np.corrcoef(M1[:,g].ravel(), M2[:,g].ravel())[0,1]:.2f} PC1 scores {U[:,0]*s[0]}")
for h in range(16):
    g = h // 4; v1 = np.linalg.svd(Mc[:, g], full_matrices=False)[2][0]
    full = (qlast[:, h] @ Mc[:, g].T) / np.sqrt(128); part = (qlast[:, h] @ np.outer(v1, v1) @ Mc[:, g].T) / np.sqrt(128)
    om = np.mean([A[off[i]+spans[i][0][0]:off[i]+spans[i][9][1], h].sum() for i in range(120)])
    print(f"L20h{h:2d}: opt-mass {om:.2f} slot score {full.mean(0)} rank1 range {np.ptp(part.mean(0)):.2f} sd {full.std(0).mean():.2f} qfrac {np.mean(np.abs(qlast[:,h]@v1))/np.linalg.norm(qlast[:,h],axis=1).mean():.2f}")
