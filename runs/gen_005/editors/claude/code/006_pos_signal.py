# Pass 2 analysis: is "how far into the trace" linearly readable from the (RMS-normalised) residual?
# Direction = mean(late, >2200 generated tokens) - mean(mid, 500-1500), leave-one-trace-out.
import json, numpy as np
from wsp.editor import *
M = json.load(open("work/pos_meta.json")); F = "work/fwd_02"
off = np.load(f"{F}/offsets.npy"); print("offsets", off.shape, off[:3])
nl = set(i for i in range(tok.vocab_size) if "\n\n" in tok.decode([i]))
res = {}
for L in (14, 20):
    H = np.load(f"{F}/h{L}.npy", mmap_mode="r")
    X, T, K, NL = [], [], [], []
    for k, q in enumerate(M["ids"]):
        a = off[k]; b = off[k + 1] if len(off) > len(M["ids"]) else a + M["n"][k]
        h = np.asarray(H[a:b], dtype=np.float32); h = h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-6)
        pl = M["prompt_len"][k]; t = np.arange(b - a) - pl
        sel = np.arange(pl + 50, b - a, 4)                      # subsample generated positions
        X.append(h[sel]); T.append(t[sel]); K.append(np.full(len(sel), k))
    X = np.concatenate(X); T = np.concatenate(T); K = np.concatenate(K)
    S = np.zeros(len(T)); aucs = []
    for k in range(len(M["ids"])):
        tr = K != k
        d = X[tr & (T > 2200)].mean(0) - X[tr & (T >= 500) & (T <= 1500)].mean(0)
        mu = X[tr].mean(0)
        S[~tr] = (X[~tr] - mu) @ d / np.linalg.norm(d)
    bins = [0, 250, 500, 1000, 1500, 2000, 2200, 2400, 2700, 3100]
    prof = [float(S[(T >= lo) & (T < hi)].mean()) if ((T >= lo) & (T < hi)).any() else None for lo, hi in zip(bins, bins[1:])]
    late, mid = S[T > 2200], S[(T >= 500) & (T <= 1500)]
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(np.r_[np.ones(len(late)), np.zeros(len(mid))], np.r_[late, mid])
    # per-trace: first generated position where score exceeds a threshold set at the 95th pct of mid scores
    th = np.percentile(mid, 95)
    first = [int(T[(K == k) & (S > th)].min()) if ((K == k) & (S > th)).any() else None for k in range(len(M["ids"]))]
    print(f"L{L}: held-out AUC late vs mid {auc:.3f}; corr(score, t) {np.corrcoef(S, T)[0,1]:.3f}")
    print("   profile by bin", [f"{lo}-{hi}:{p:.1f}" for lo, hi, p in zip(bins, bins[1:], prof) if p is not None])
    print("   first crossing of mid-p95 per trace", first)
    res[L] = dict(auc=auc, prof=prof, first=first)
json.dump(res, open("work/pos_signal.json", "w"), default=float, indent=1)
