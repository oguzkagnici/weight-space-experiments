# The pooled "late" direction (code/007) fires near the END of finished traces, never in cut traces: it reads
# "wrapping up", not length. Refit on cut traces only (late > 2200 vs mid 500-1500, leave-one-cut-trace-out),
# score finished traces with the direction from all 7 cut traces.
import json, numpy as np
from wsp.editor import *
M = json.load(open("work/pos_meta.json")); F = "work/fwd_02"; off = np.load(f"{F}/offsets.npy")
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
NC = 7
for L in (14, 20):
    H = np.load(f"{F}/h{L}.npy", mmap_mode="r"); X, T, K = [], [], []
    for k, q in enumerate(M["ids"]):
        ids = tok(trace_text(B[q]), add_special_tokens=False).input_ids; pl = M["prompt_len"][k]
        br = [i for i in range(pl + 20, len(ids)) if "\n\n" in tok.decode([ids[i]])]
        h = np.asarray(H[off[k] + np.array(br)], dtype=np.float32); h /= np.sqrt((h ** 2).mean(1, keepdims=True))
        X.append(h); T.append(np.array(br) - pl); K.append(np.full(len(br), k))
    X = np.concatenate(X); T = np.concatenate(T); K = np.concatenate(K); S = np.zeros(len(T))
    cut = K < NC
    for k in range(len(M["ids"])):
        tr = cut & (K != k)
        d = X[tr & (T > 2200)].mean(0) - X[tr & (T >= 500) & (T <= 1500)].mean(0)
        S[K == k] = (X[K == k] - X[tr].mean(0)) @ d / np.linalg.norm(d)
    print(f"L{L}")
    for nm, m0 in (("cut (held out)", cut), ("finished", ~cut)):
        print("  ", nm, " ".join(f"{lo}-{hi}:{S[m0 & (T >= lo) & (T < hi)].mean():5.2f}" for lo, hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3100)] if (m0 & (T >= lo) & (T < hi)).any()))
    for pct in (99, 100):
        th = np.percentile(S[T < 1500], pct)
        first = [int(T[(K == k) & (S > th)].min()) if ((K == k) & (S > th)).any() else None for k in range(len(M["ids"]))]
        print(f"   th p{pct}<1500 = {th:.1f}: first fire", first)
