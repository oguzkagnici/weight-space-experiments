# hypothesis H6: a correct-vs-incorrect "belief" at option-end tokens is written mainly by a few blocks in layers 8-14;
# measure, with 2-fold cross-validation over questions, (a) the probe accuracy (pick the option whose end-token residual
# projects highest on the fold's correct-minus-incorrect mean direction) per layer and (b) per block delta.
import json, numpy as np
from wsp.editor import *
d = json.load(open("work/option_ends.json")); qs = d["questions"]
off = np.load("work/fwd_01/offsets.npy")
H = {l: np.load(f"work/fwd_01/h{l}.npy", mmap_mode="r") for l in range(7, 16)}
rng = np.random.default_rng(0); fold = rng.permutation(len(qs)) % 2
def feats(arr):
    X, y, g = [], [], []
    for i, q in enumerate(qs):
        e = np.array(d["ends"][q]["ends"]) + off[i]
        v = arr[e].astype(np.float64); v = v - v.mean(0)            # centre within question (relative belief)
        X.append(v); y += [int(j == d["ends"][q]["answer"]) for j in range(len(e))]; g += [i] * len(e)
    return np.concatenate(X), np.array(y), np.array(g)
def probe_acc(X, y, g):
    accs = []
    for f in (0, 1):
        tr = fold[g] != f; te = ~tr
        w = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0); w /= np.linalg.norm(w)
        s = X[te] @ w
        for i in np.unique(g[te]):
            m = g == i; accs.append(int(np.argmax(s[m[te]]) == np.argmax(y[m])))
    return np.mean(accs), w
chance = np.mean([1 / len(d["ends"][q]["ends"]) for q in qs])
print(f"chance {chance:.3f}")
print("layer | residual probe acc | block (h_l - h_l-1) probe acc")
prev = None
for l in range(7, 16):
    X, y, g = feats(H[l]); a, _ = probe_acc(X, y, g)
    b = ""
    if l > 7:
        Xd, _, _ = feats(np.asarray(H[l]) - np.asarray(H[l - 1])); ab, _ = probe_acc(Xd, y, g); b = f"{ab:.3f}"
    print(f"  h{l:2d} | {a:.3f} | {b}")
