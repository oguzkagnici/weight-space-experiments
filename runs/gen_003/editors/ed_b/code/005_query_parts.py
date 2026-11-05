# Hypothesis: the H12 query = constant part (q_bar) + question-specific part (dq). If dq carries content matching, amplifying it (q_bar + gamma*dq, renormalized) raises H12's gold pick rate; if q_bar alone does as well, content lives only in keys.
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
H = 12; G = 3
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    data.append(dict(K=np.asarray(Kn[a:b, G]), q=np.asarray(Qn[b - 1, H]), toks=[list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]], gold=o["gold"], n=o["n"]))
n = len(data)
Qall = np.array([d["q"] for d in data])
qbar = Qall.mean(0)
print("|qbar| %.2f  mean|q| %.2f  mean|dq| %.2f  cos(q,qbar) mean %.3f" % (np.linalg.norm(qbar), np.linalg.norm(Qall, axis=1).mean(), np.linalg.norm(Qall - qbar, axis=1).mean(), np.mean(Qall @ qbar / np.linalg.norm(Qall, axis=1) / np.linalg.norm(qbar))))

def evalq(fn):
    acc, lpg, rank = [], [], []
    for i, d in enumerate(data):
        qq = fn(i, d)
        lg = d["K"] @ qq / np.sqrt(128)
        s = np.array([np.logaddexp.reduce(lg[t]) for t in d["toks"]])
        acc.append(np.argmax(s) == d["gold"]); lpg.append(s[d["gold"]] - np.logaddexp.reduce(s)); rank.append((s > s[d["gold"]]).sum())
    return np.mean(acc), np.mean(lpg), np.mean(rank)

print("actual q      acc %.3f lshare %.3f rank %.2f" % evalq(lambda i, d: d["q"]))
# leave-one-out mean query
print("constant q    acc %.3f lshare %.3f rank %.2f" % evalq(lambda i, d: (qbar * n - d["q"]) / (n - 1)))
for gam in [0.0, 0.5, 1.5, 2.0, 3.0]:
    def f(i, d, gam=gam):
        qb = (qbar * n - d["q"]) / (n - 1)
        qq = qb + gam * (d["q"] - qb)
        return qq * np.linalg.norm(d["q"]) / np.linalg.norm(qq)
    print("gamma %.1f      acc %.3f lshare %.3f rank %.2f" % ((gam,) + evalq(f)))
# temperature only (scale whole q): sharpening the same ranking
for t in [1.5, 2.0]:
    print("q*%.1f         acc %.3f lshare %.3f rank %.2f" % ((t,) + evalq(lambda i, d, t=t: d["q"] * t)))
# PCA of dq: how many dims
U, S, Vt = np.linalg.svd(Qall - qbar, full_matrices=False)
print("dq variance explained by top PCs:", np.round((S ** 2 / (S ** 2).sum())[:8], 3))
np.save("work/qbar_h12.npy", qbar)
