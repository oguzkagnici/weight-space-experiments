# Is the length score cleaner at paragraph breaks ("\n\n" tokens), where the edit would fire?
import json, numpy as np
from sklearn.metrics import roc_auc_score
from wsp.editor import *
M = json.load(open("work/pos_meta.json")); F = "work/fwd_02"; off = np.load(f"{F}/offsets.npy")
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
for L in (14, 20):
    H = np.load(f"{F}/h{L}.npy", mmap_mode="r"); X, T, K = [], [], []
    for k, q in enumerate(M["ids"]):
        ids = tok(trace_text(B[q]), add_special_tokens=False).input_ids
        assert len(ids) == off[k + 1] - off[k], (len(ids), off[k + 1] - off[k])
        pl = M["prompt_len"][k]
        br = [i for i in range(pl + 20, len(ids)) if "\n\n" in tok.decode([ids[i]])]
        h = np.asarray(H[off[k] + np.array(br)], dtype=np.float32); h /= np.sqrt((h ** 2).mean(1, keepdims=True))
        X.append(h); T.append(np.array(br) - pl); K.append(np.full(len(br), k))
    X = np.concatenate(X); T = np.concatenate(T); K = np.concatenate(K); S = np.zeros(len(T))
    for k in range(len(M["ids"])):
        tr = K != k
        d = X[tr & (T > 2200)].mean(0) - X[tr & (T >= 500) & (T <= 1500)].mean(0)
        S[~tr] = (X[~tr] - X[tr].mean(0)) @ d / np.linalg.norm(d)
    late, mid = S[T > 2200], S[(T >= 500) & (T <= 1500)]
    print(f"L{L}: breaks n={len(T)}, AUC late vs mid {roc_auc_score(np.r_[np.ones(len(late)), np.zeros(len(mid))], np.r_[late, mid]):.3f}, corr {np.corrcoef(S, T)[0,1]:.2f}")
    for lo, hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2200), (2200, 2500), (2500, 3100)]:
        m = (T >= lo) & (T < hi); print(f"   {lo}-{hi}: n={m.sum():3d} mean {S[m].mean():5.2f} sd {S[m].std():4.2f}")
    print("   per-trace first firing at break (leave-one-out score):")
    for pct in (99, 99.5, 100):
        th = np.percentile(S[T < 1500], pct)
        first = [int(T[(K == k) & (S > th)].min()) if ((K == k) & (S > th)).any() else None for k in range(len(M["ids"]))]
        print(f"   th={th:.1f} (p{pct} of <1500): late fire rate {np.mean(S[T > 2200] > th):.2f}; first fire", first)
