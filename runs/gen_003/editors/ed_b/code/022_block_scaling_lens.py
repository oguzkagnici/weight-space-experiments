# Hypothesis: some layer blocks write the option-end belief more reliably than others; in the key lens (keys from h22 + (f-1)*(h_b - h_a)), scaling a block's residual contribution at option tokens by f>1 would raise H12's gold pick if that block carries net-correct evidence, and lower it if it carries distractor evidence. DEV[:120].
import numpy as np, pickle
from wsp.editor import load
D = "work/fwd_03/"
sp = pickle.load(open("work/spans240.pkl", "rb"))[:120]
off = np.load(D + "offsets.npy")
qlast = np.load("work/p1_q_last.npy")[:120, 12].astype(np.float64)
gop = load("model.layers.22.operator_norm.weight").numpy().astype(np.float64)
Wk = load("model.layers.22.self_attn.k_proj.weight")[3 * 128:4 * 128].numpy().astype(np.float64)
gk = load("model.layers.22.self_attn.k_norm.weight").numpy().astype(np.float64)
def keys(h):
    xn = gop * h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-5)
    y = xn @ Wk.T
    return gk * y / np.sqrt((y ** 2).mean(1, keepdims=True) + 1e-5)
H = {l: np.load(D + "h%d.npy" % l, mmap_mode="r") for l in [10, 12, 14, 16, 18, 20, 22]}
blocks = [(10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20, 22)]
def run(fn):
    acc = []; ls = []
    for i, o in enumerate(sp):
        a, b = off[i], off[i + 1]
        h = fn(a, b)
        lg = keys(h) @ qlast[i] / np.sqrt(128)
        s = np.array([np.logaddexp.reduce(lg[s0:s1 + 2]) for (s0, s1) in o["spans"]])
        acc.append(np.argmax(s) == o["gold"]); ls.append(s[o["gold"]] - np.logaddexp.reduce(s))
    return np.array(acc), np.mean(ls)
a0, l0 = run(lambda a, b: np.asarray(H[22][a:b]).astype(np.float64))
print("base acc %.3f ls %.3f" % (a0.mean(), l0))
for (la, lb) in blocks:
    out = []
    for f in [0.0, 0.5, 1.5, 2.0]:
        def fn(a, b, f=f):
            h22 = np.asarray(H[22][a:b]).astype(np.float64)
            c = np.asarray(H[lb][a:b]).astype(np.float64) - np.asarray(H[la][a:b]).astype(np.float64)
            return h22 + (f - 1) * c
        ac, ls = run(fn)
        out.append("f%.1f: %.3f (+%d/-%d) ls %+.3f" % (f, ac.mean(), (ac & ~a0).sum(), (~ac & a0).sum(), ls - l0))
    print("block h%d->h%d | " % (la, lb) + " | ".join(out))
