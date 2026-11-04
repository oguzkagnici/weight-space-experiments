# Hypothesis: the early-option preference of L22 heads is a slot/letter effect (A,B keys) rather than a token-distance (RoPE) effect.
import json, numpy as np
from wsp.editor import *
sp = json.load(open("work/split.json")); ids = sp["dev"][:512]; Q = questions()
spans = json.load(open("work/p1_spans.json"))
otok = np.load("work/fwd_01/offsets_tok.npy")
A = np.load("work/fwd_01/attn_last_L22.npy")
rows = []
for qi in range(512):
    if len(spans[qi]) != 10: continue
    n = otok[qi+1]-otok[qi]; a = A[otok[qi]:otok[qi+1]]
    for k, (t0, t1, tl) in enumerate(spans[qi]):
        for t in range(t0, t1):
            rows.append((qi, k, (n-1-t)/100.0, t == tl, t - t0, np.log(max(float(a[t]) if False else 0,0)+1e-9) if False else np.log(np.maximum(a[t].astype(np.float32), 1e-7))))
qi_ = np.array([r[0] for r in rows]); k_ = np.array([r[1] for r in rows]); d_ = np.array([r[2] for r in rows])
islet = np.array([r[3] for r in rows]).astype(float); rel = np.array([r[4] for r in rows])
Y = np.stack([r[5] for r in rows])  # ntok x 16
# demean per question
uq, inv = np.unique(qi_, return_inverse=True)
def dm(v):
    s = np.zeros((len(uq),) + v.shape[1:]); c = np.bincount(inv)
    np.add.at(s, inv, v); return v - (s / c.reshape(-1, *([1]*(v.ndim-1))))[inv]
X = np.column_stack([np.eye(10)[k_][:, 1:], d_, islet, (rel == 0), (rel == 2), np.minimum(rel, 20)/20]).astype(float)
Xd = dm(X); Yd = dm(Y)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
B = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
print("rows:", len(rows), " coef columns: slots B..J (vs A), dist/100tok, is-letter-tok, rel0 '(' , rel2 ')' , rel/20")
for h in [0, 3, 8, 10, 11, 12, 13, 14, 15]:
    print(f"L22h{h:2d}", B[:, h])
# distance-only model on slot-free data: within-slot correlation
print("corr(slot index, distance) =", np.corrcoef(k_, d_)[0,1])
