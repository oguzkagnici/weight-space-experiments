# Hypothesis: H12's normalized query has components in directions where the effective keys (g_q*k) barely vary across tokens; they add a constant logit and waste the fixed query norm. Projecting q_proj (head 12) onto the top-r within-text key-variance subspace is a pure per-head temperature increase (q_norm rescales the rest) -> lens: sharper H12, less long-option bias, higher gold pick.
import numpy as np, pickle
from wsp.editor import load
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
gq = load("model.layers.22.self_attn.q_norm.weight").numpy().astype(np.float64)
H = 12; G = 3
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    K = np.asarray(Kn[a:b, G]).astype(np.float64); q = np.asarray(Qn[b - 1, H]).astype(np.float64)
    data.append(dict(K=K, y=q / gq, toks=[list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]], gold=o["gold"], n=o["n"]))
n = len(data)
def cov(idx):
    C = np.zeros((128, 128)); m = 0
    for i in idx:
        kap = gq * data[i]["K"]
        kc = kap - kap.mean(0)
        C += kc.T @ kc; m += len(kc)
    return C / m
def evaluate(fn):
    acc, ls, mass, picks, scale = [], [], [], [], []
    for i, d in enumerate(data):
        y = fn(i, d)
        sc = np.linalg.norm(d["y"]) / np.linalg.norm(y); y = y * sc
        lg = d["K"] @ (gq * y) / np.sqrt(128)
        z = np.logaddexp.reduce(lg)
        s = np.array([np.logaddexp.reduce(lg[t]) for t in d["toks"]])
        acc.append(np.argmax(s) == d["gold"]); ls.append(s[d["gold"]] - np.logaddexp.reduce(s)); mass.append(np.exp(np.logaddexp.reduce(s) - z)); scale.append(sc)
    return np.array(acc), np.mean(ls), np.mean(mass), np.mean(scale)
a0, l0, m0, _ = evaluate(lambda i, d: d["y"])
print("orig: acc %.3f lshare %.3f optmass %.3f" % (a0.mean(), l0, m0))
C = cov(range(n)); ev, V = np.linalg.eigh(C); ev = ev[::-1]; V = V[:, ::-1]
cum = np.cumsum(ev) / ev.sum()
print("key-var cumulative at r=8,16,32,64,96:", np.round(cum[[7, 15, 31, 63, 95]], 3))
Y = np.array([d["y"] for d in data]); Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
for r in [8, 16, 32, 64, 96]:
    print("r=%d: mean fraction of |y|^2 in top-r key subspace %.3f" % (r, np.mean(((Yn @ V[:, :r]) ** 2).sum(1))))
folds = [list(range(0, n, 2)), list(range(1, n, 2))]
Vs = []
for f in range(2):
    e, v = np.linalg.eigh(cov(folds[1 - f])); Vs.append(v[:, ::-1])
for r in [8, 16, 24, 32, 48, 64, 96]:
    for al in [1.0]:
        def fn(i, d, r=r, al=al):
            V_ = Vs[i % 2][:, r:]  # discard low-variance directions
            return d["y"] - al * V_ @ (V_.T @ d["y"])
        a, l, m, sc = evaluate(fn)
        print("keep top-%d (alpha %.1f): temp-gain %.2f acc %.3f (w->r %d r->w %d) lshare %.3f optmass %.3f" % (r, al, sc, a.mean(), (a & ~a0).sum(), (~a & a0).sum(), l, m))
