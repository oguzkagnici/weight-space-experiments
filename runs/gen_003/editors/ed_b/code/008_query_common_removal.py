# Hypothesis: part of H12's (RMS-normed) query is spent on the common key direction (constant logit over tokens). Removing that component from q_proj rows of head 12 (rank-1, W <- (I - a w w^T) W) lets q_norm rescale the rest, sharpening H12 per-head (lens: T~2 gave +2.5 pts H12 pick, less long-option bias). Lens pre-screen with CV.
import numpy as np, pickle
from wsp.editor import load
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
Aatt = np.load(D + "attn_last_L22.npy")
gq = load("model.layers.22.self_attn.q_norm.weight").numpy().astype(np.float64)
gk = load("model.layers.22.self_attn.k_norm.weight").numpy().astype(np.float64)
print("q_norm w: min %.3f max %.3f mean %.3f | k_norm w: min %.3f max %.3f mean %.3f" % (gq.min(), gq.max(), gq.mean(), gk.min(), gk.max(), gk.mean()))
H = 12; G = 3
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    K = np.asarray(Kn[a:b, G]).astype(np.float64); q = np.asarray(Qn[b - 1, H]).astype(np.float64)
    data.append(dict(K=K, y=q / gq, toks=[list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]], gold=o["gold"], n=o["n"], att=Aatt[a:b, H]))
n = len(data)
# check rms of y ~ 1
print("rms(y) mean %.3f" % np.mean([np.sqrt((d["y"] ** 2).mean()) for d in data]))

def evaluate(fn):
    acc, ls, mass, picks = [], [], [], []
    for i, d in enumerate(data):
        y = fn(i, d)
        y = y / np.sqrt((y ** 2).mean())
        lg = d["K"] @ (gq * y) / np.sqrt(128)
        z = np.logaddexp.reduce(lg)
        s = np.array([np.logaddexp.reduce(lg[t]) for t in d["toks"]])
        acc.append(np.argmax(s) == d["gold"]); ls.append(s[d["gold"]] - np.logaddexp.reduce(s)); mass.append(np.exp(np.logaddexp.reduce(s) - z)); picks.append(np.argmax(s))
    return np.array(acc), np.mean(ls), np.mean(mass), np.array(picks)

def common_dir(idx, which):
    acc = []
    for i in idx:
        d = data[i]
        kap = gq * d["K"]
        if which == "all": acc.append(kap.mean(0))
        elif which == "opt": acc.append(kap[sum(d["toks"], [])].mean(0))
        elif which == "attw": acc.append(d["att"] @ kap)
    w = np.mean(acc, 0); return w / np.linalg.norm(w)

a0, l0, m0, p0 = evaluate(lambda i, d: d["y"])
print("orig: acc %.3f lshare %.3f optmass %.3f" % (a0.mean(), l0, m0))
folds = [list(range(0, n, 2)), list(range(1, n, 2))]
for which in ["all", "opt", "attw"]:
    W = [common_dir(folds[1 - f], which) for f in range(2)]
    frac = np.mean([abs(d["y"] @ W[i % 2]) / np.linalg.norm(d["y"]) for i, d in enumerate(data)])
    for al in [0.5, 0.75, 0.9, 1.0]:
        def fn(i, d, al=al):
            w = W[i % 2]; return d["y"] - al * (d["y"] @ w) * w
        a, l, m, p = evaluate(fn)
        print("%s frac|y.w|/|y| %.2f alpha %.2f: acc %.3f (w->r %d r->w %d) lshare %.3f optmass %.3f" % (which, frac, al, a.mean(), (a & ~a0).sum(), (~a & a0).sum(), l, m))
