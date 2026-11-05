# Hypothesis: H12 puts ~12% of its attention on the option letter tokens "(X", whose keys cannot contain the option's content (they precede it, causal), so that attention is a content-free slot vote. Removing it (lens: drop letter tokens; edit: project the letter-token-vs-end-token key direction out of the H12 query, renormalized) should raise H12's gold pick and flatten its B/C preference.
import numpy as np, pickle
from wsp.editor import load
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
Q = np.load("work/p1_q_last.npy")[:, 12].astype(np.float64)
K = np.load("work/p1_k_g3.npy").astype(np.float64)
gq = load("model.layers.22.q_norm.weight".replace("q_norm", "self_attn.q_norm")).numpy().astype(np.float64)
n = len(sp)
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    data.append(dict(K=K[a:b], q=Q[i], spans=o["spans"], gold=o["gold"], n=o["n"]))
def scores(q, d, drop_letter=False, types=None):
    lg = d["K"] @ q / np.sqrt(128)
    out = []
    for (s0, s1) in d["spans"]:
        t = list(range(s0 + (1 if drop_letter else 0), s1 + 2))
        out.append(np.logaddexp.reduce(lg[t]))
    return np.array(out), lg
def evaluate(fn, drop=False):
    acc, cnt, lshare, lettmass = [], np.zeros(10), [], []
    for i, d in enumerate(data):
        q = fn(i, d)
        s, lg = scores(q, d, drop)
        p = np.argmax(s); acc.append(p == d["gold"]); lshare.append(s[d["gold"]] - np.logaddexp.reduce(s))
        if d["n"] == 10: cnt[p] += 1
        z = np.logaddexp.reduce(lg); lettmass.append(sum(np.exp(lg[s0] - z) for (s0, s1) in d["spans"]))
    return np.array(acc), cnt.astype(int), np.mean(lshare), np.mean(lettmass)
a0, c0, l0, m0 = evaluate(lambda i, d: d["q"])
print("orig: acc %.3f ls %.3f letter-token mass %.3f counts %s" % (a0.mean(), l0, m0, c0))
a1, c1, l1, _ = evaluate(lambda i, d: d["q"], drop=True)
print("drop letter tokens from option scores: acc %.3f (+%d/-%d) counts %s" % (a1.mean(), (a1 & ~a0).sum(), (~a1 & a0).sum(), c1))
# direction: mean over questions of (mean letter-token key - mean end/newline key), in q (=effective) space
def udir(idx):
    acc = []
    for i in idx:
        d = data[i]
        let = d["K"][[s0 for (s0, s1) in d["spans"]]].mean(0)
        end = d["K"][[s1 + 1 for (s0, s1) in d["spans"]] + [s1 for (s0, s1) in d["spans"]]].mean(0)
        acc.append(let - end)
    u = np.mean(acc, 0); return u / np.linalg.norm(u)
folds = [list(range(0, n, 2)), list(range(1, n, 2))]
U = [udir(folds[1 - f]) for f in range(2)]
print("mean q.u %.2f (|q| %.1f)" % (np.mean([d["q"] @ U[i % 2] for i, d in enumerate(data)]), np.mean(np.linalg.norm(Q, axis=1))))
for al in [0.5, 1.0, 1.5, 2.0]:
    def fn(i, d, al=al):
        # remove in normalized-query space: q = g*y; y' = y - al (y.w) w, with w ~ g*u direction
        y = d["q"] / gq; w = gq * U[i % 2]; w /= np.linalg.norm(w)
        y2 = y - al * (y @ w) * w; y2 *= np.linalg.norm(y) / np.linalg.norm(y2)
        return gq * y2
    a, c, l, m = evaluate(fn)
    print("project letter dir alpha %.1f: acc %.3f (+%d/-%d) ls %.3f letter mass %.3f counts %s" % (al, a.mean(), (a & ~a0).sum(), (~a & a0).sum(), l, m, c))
