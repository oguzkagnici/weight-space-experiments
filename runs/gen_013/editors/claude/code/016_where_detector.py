# Where along a corrupted text does the detector score (x.w_in / x.p) differ from the clean text, and how does it rank among background positions?
import json, numpy as np
from wsp.editor import *
D = "work/fwd_04/"; ix = json.load(open("work/read4_index.json")); m = json.load(open(D + "meta.json"))
nt = json.loads(m["n_tokens"]) if isinstance(m["n_tokens"], str) else m["n_tokens"]; off = np.concatenate([[0], np.cumsum(nt)])
X = np.load(D + "in_model.layers.23.feed_forward.w1.npy", mmap_mode="r"); w_in, p = np.load("work/gate_vectors.npy")
sc = lambda Z: (Z @ w_in) / (Z @ p)
bg = np.concatenate([np.arange(off[k] + 4, off[k + 1]) for k, it in enumerate(ix) if it["kind"] == "bank"])
sB = sc(np.asarray(X[bg], np.float64)); pct = lambda v: float(np.mean(sB < v))
for k in range(0, len(ix), 2):
    if ix[k]["kind"] != "pair": continue
    a = sc(np.asarray(X[off[k]:off[k + 1]], np.float64)); b = sc(np.asarray(X[off[k + 1]:off[k + 2]], np.float64))
    L = min(len(a), len(b)); d0 = next((i for i in range(L) if abs(a[i] - b[i]) > 1e-6), L)
    tail = b[d0:] - a[d0:len(b)] if len(a) >= len(b) else b[d0:len(a)] - a[d0:]
    j = int(np.argmax(b[d0:])) + d0
    print(f"pair {ix[k]['i']:3d} first diff tok {d0}/{len(b)}; corrupt max score after diff at +{j-d0} (bg pct {pct(b[j]):.4f}); end: corrupt {pct(b[-1]):.3f} clean {pct(a[-1]):.3f}; mean diff after {np.mean(tail):+.3f}")
