# Hypothesis: L22 group-3 keys at option tokens carry a correct-option direction d (gold-minus-wrong mean) that H12's query only partly uses; adding beta*d to the H12 query (renormalized) would raise H12's gold pick rate in cross-validation (lens-level pre-screen on recorded q,k).
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
A = np.load(D + "attn_last_L22.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
H = 12; G = 3
# attention by token type within option spans
tt = np.zeros(5)  # letter, content(not last), last content, newline, outside
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]; att = A[a:b, H]
    inside = np.zeros(o["T"], bool)
    for (s0, s1) in o["spans"]:
        tt[0] += att[s0]; tt[1] += att[s0 + 1:s1].sum(); tt[2] += att[s1]; tt[3] += att[s1 + 1]
        inside[s0:s1 + 2] = True
    tt[4] += att[~inside].sum()
print("H12 attention share: letter %.3f content %.3f lastcontent %.3f newline %.3f outside %.3f" % tuple(tt / tt.sum()))

# per question: keys of option tokens (last content and newline), q at last
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    K = np.asarray(Kn[a:b, G]); q = np.asarray(Qn[b - 1, H])
    toks = [list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]]
    data.append(dict(K=K, q=q, toks=toks, gold=o["gold"], n=o["n"], spans=o["spans"]))

def optscore(qv, d):
    lg = d["K"] @ qv / np.sqrt(128)
    return np.array([np.logaddexp.reduce(lg[t]) for t in d["toks"]])

def gold_dir(ds, kind):
    diffs = []
    for d in ds:
        idx = [ (s1 if kind == "last" else s1 + 1 if kind == "nl" else s0) for (s0, s1) in d["spans"]]
        Kk = d["K"][idx]
        g = d["gold"]; w = [j for j in range(d["n"]) if j != g]
        diffs.append(Kk[g] - Kk[w].mean(0))
    return np.mean(diffs, 0), np.array(diffs)

n = len(data)
for kind in ["last", "nl", "letter"]:
    m, df = gold_dir(data, kind)
    snr = (df @ m / np.linalg.norm(m)).mean() / (df @ m / np.linalg.norm(m)).std()
    print(kind, "|d| %.2f  proj snr %.2f  q.d_hat mean %.2f  |q| %.1f" % (np.linalg.norm(m), snr, np.mean([d["q"] @ m / np.linalg.norm(m) for d in data]), np.mean([np.linalg.norm(d["q"]) for d in data])))

# CV: 2 folds, direction from one fold, apply on the other
folds = [list(range(0, n, 2)), list(range(1, n, 2))]
for kind in ["last", "nl"]:
    for beta in [0, 2, 4, 8, 16]:
        acc = []; lpg = []
        for f in range(2):
            m, _ = gold_dir([data[i] for i in folds[1 - f]], kind)
            m = m / np.linalg.norm(m)
            for i in folds[f]:
                d = data[i]; q = d["q"]; qq = q + beta * m
                qq = qq * np.linalg.norm(q) / np.linalg.norm(qq)
                s = optscore(qq, d)
                acc.append(np.argmax(s) == d["gold"]); lpg.append(s[d["gold"]] - np.logaddexp.reduce(s))
        print("CV %s beta %4.1f: H12 option-argmax acc %.3f  mean log share gold %.3f" % (kind, beta, np.mean(acc), np.mean(lpg)))
