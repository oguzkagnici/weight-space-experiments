# Hypothesis: at option tokens, a middle-layer residual may already hold a better correctness belief than h22 and later layers lose it ("knows then loses" at option positions). Key lens: read group-3 keys from h_l through L22 operator_norm, k_proj, k_norm; score options with H12's recorded query; compare gold-pick per layer (DEV[:120]).
import numpy as np, pickle
from wsp.editor import load
D = "work/fwd_03/"
import json
meta = json.load(open(D + "meta.json")) if __import__("os").path.exists(D + "meta.json") else None
sp = pickle.load(open("work/spans240.pkl", "rb"))[:120]
off = np.load(D + "offsets.npy")
qlast = np.load("work/p1_q_last.npy")[:120, 12].astype(np.float64)
Krec = np.load("work/p1_k_g3.npy", mmap_mode="r")
off1 = np.load("work/fwd_01/offsets_tok.npy")
gop = load("model.layers.22.operator_norm.weight").numpy().astype(np.float64)
Wk = load("model.layers.22.self_attn.k_proj.weight")[3 * 128:4 * 128].numpy().astype(np.float64)
gk = load("model.layers.22.self_attn.k_norm.weight").numpy().astype(np.float64)
def keys(h):
    xn = gop * h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-5)
    y = xn @ Wk.T
    return gk * y / np.sqrt((y ** 2).mean(1, keepdims=True) + 1e-5)
res = {}
for l in [10, 12, 14, 16, 18, 20, 22]:
    Hs = np.load(D + "hidden_states_%d.npy" % l, mmap_mode="r") if __import__("os").path.exists(D + "hidden_states_%d.npy" % l) else np.load(D + "h%d.npy" % l, mmap_mode="r")
    acc, ls, err = [], [], []
    for i, o in enumerate(sp):
        a, b = off[i], off[i + 1]
        K = keys(np.asarray(Hs[a:b]).astype(np.float64))
        if l == 22:
            err.append(np.abs(K - np.asarray(Krec[off1[i]:off1[i + 1]])).max())
        lg = K @ qlast[i] / np.sqrt(128)
        s = np.array([np.logaddexp.reduce(lg[s0:s1 + 2]) for (s0, s1) in o["spans"]])
        acc.append(np.argmax(s) == o["gold"]); ls.append(s[o["gold"]] - np.logaddexp.reduce(s))
    res[l] = (np.array(acc), np.mean(ls))
    print("keys from h%d: H12 gold pick %.3f  log share gold %.3f %s" % (l, np.mean(acc), np.mean(ls), ("(max |k - recorded| %.3f)" % max(err)) if err else ""))
a22 = res[22][0]
for l in res:
    print("h%d vs h22: right only at h%d: %d, right only at h22: %d" % (l, l, (res[l][0] & ~a22).sum(), (~res[l][0] & a22).sum()))
