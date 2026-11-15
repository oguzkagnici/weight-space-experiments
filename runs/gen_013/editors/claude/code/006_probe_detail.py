# Is the corrupt-vs-clean direction general (fit on one kind, test on the other) and is it more than surprisal? Unpaired AUC (gate quality).
import json, numpy as np
from sklearn.metrics import roc_auc_score
P = json.load(open("work/pairs1.json")); kinds = np.array([p["kind"] for p in P]); n = len(P)
S = np.load("work/surpr1.npy"); ds = S[1] - S[0]
D = "work/fwd_01/"
for L in [8, 12, 14, 18, 20, 24]:
    H = np.load(D + f"h{L}.npy"); C, X = H[0::2], H[1::2]; d = X - C
    out = []
    for a, b in (("arith", "swap"), ("swap", "arith")):
        w = d[kinds == a].mean(0); m = kinds == b
        out.append(f"fit {a}->test {b}: paired {np.mean((X[m]@w) > (C[m]@w)):.2f} AUC {roc_auc_score(np.r_[np.zeros(m.sum()), np.ones(m.sum())], np.r_[C[m]@w, X[m]@w]):.2f}")
    # split-half on swaps by question for unpaired AUC
    qs = sorted({p['id'] for p in P}); rng = np.random.default_rng(1); au = []; cs = []
    for r in range(20):
        tr = set(rng.choice(qs, len(qs)//2, replace=False)); mt = np.array([p['id'] in tr for p in P])
        w = d[mt].mean(0); te = ~mt
        sc_c, sc_x = C[te]@w, X[te]@w
        au.append(roc_auc_score(np.r_[np.zeros(te.sum()), np.ones(te.sum())], np.r_[sc_c, sc_x]))
        cs.append(np.corrcoef(sc_x - sc_c, ds[te])[0, 1])
    print(f"h{L}: " + " | ".join(out) + f" | all split-half unpaired AUC {np.mean(au):.2f}; corr(score diff, logprob diff) {np.mean(cs):+.2f}")
