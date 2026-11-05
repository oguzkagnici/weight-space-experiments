# Hypothesis: some directions of the H12 query (in the eigenbasis of the effective-key covariance) read key features that systematically favour wrong options (position, length, surface); removing those from q_proj head 12 (projection, rank-k) raises H12's gold pick in cross-validation (lens on recorded q,k).
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
    toks = [list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]]
    data.append(dict(K=K, y=q / np.where(np.abs(gq) < 1e-3, 1e-3, gq), q=q, toks=toks, gold=o["gold"], n=o["n"]))
n = len(data)
def option_keys(d, lg):
    # attention-weighted (within option) effective key per option, using the original logits
    out = []
    for t in d["toks"]:
        w = np.exp(lg[t] - lg[t].max()); w /= w.sum(); out.append(w @ (gq * d["K"][t]))
    return np.array(out)
for d in data:
    lg = d["K"] @ d["q"] / np.sqrt(128)
    d["ok"] = option_keys(d, lg)
def basis(idx):
    C = np.zeros((128, 128))
    for i in idx:
        kap = gq * data[i]["K"]; kc = kap - kap.mean(0); C += kc.T @ kc
    e, V = np.linalg.eigh(C); return V[:, ::-1]
def contrib(idx, V):
    # mean contribution of q's component along each basis dir to (gold - mean wrong) option-key logit
    c = []
    for i in idx:
        d = data[i]; ok = d["ok"]; gg = d["gold"]; w = [j for j in range(d["n"]) if j != gg]
        diff = ok[gg] - ok[w].mean(0)
        c.append((V.T @ d["y"]) * (V.T @ diff))
    c = np.array(c); return c.mean(0), c.std(0) / np.sqrt(len(c))
def evaluate(idx, fn):
    acc = []; ls = []
    for i in idx:
        d = data[i]; y = fn(d); y = y * np.linalg.norm(d["y"]) / np.linalg.norm(y)
        lg = d["K"] @ (gq * y) / np.sqrt(128)
        s = np.array([np.logaddexp.reduce(lg[t]) for t in d["toks"]])
        acc.append(np.argmax(s) == d["gold"]); ls.append(s[d["gold"]] - np.logaddexp.reduce(s))
    return np.array(acc), np.array(ls)
folds = [list(range(0, n, 2)), list(range(1, n, 2))]
res = {}
for f in range(2):
    tr, te = folds[1 - f], folds[f]
    V = basis(tr); m, se = contrib(tr, V)
    a0, l0 = evaluate(te, lambda d: d["y"])
    for thr in [0.0, -1.0, -2.0, -3.0]:
        bad = np.where(m / (se + 1e-9) < thr)[0] if thr < 0 else np.where(m < 0)[0]
        Vb = V[:, bad]
        a, l = evaluate(te, lambda d: d["y"] - Vb @ (Vb.T @ d["y"]))
        r = res.setdefault(thr, [0, 0, 0, [], 0])
        r[0] += (a & ~a0).sum(); r[1] += (~a & a0).sum(); r[2] += len(bad); r[3].append(l.mean() - l0.mean()); r[4] += a.sum() - a0.sum()
for thr, r in res.items():
    print("remove dirs with t<%.0f (avg %d dirs): CV w->r %d r->w %d  dlshare %.3f" % (thr, r[2] // 2, r[0], r[1], np.mean(r[3])))
