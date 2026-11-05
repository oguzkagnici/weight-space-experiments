# Hypothesis: H12's soft attention sums mass over all tokens of an option, so long options collect mass (long-option bias); a sharper H12 (higher inverse temperature) would pick by best-matching token and raise gold pick rate. Lens pre-screen on recorded q,k with paired counts.
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
rows = pickle.load(open("work/diag03.pkl", "rb"))
H = 12; G = 3
data = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    K = np.asarray(Kn[a:b, G]).astype(np.float64); q = np.asarray(Qn[b - 1, H]).astype(np.float64)
    lg = K @ q / np.sqrt(128)
    toks = [list(range(s0, s1 + 2)) for (s0, s1) in o["spans"]]
    data.append(dict(lg=lg, toks=toks, gold=o["gold"], n=o["n"], L=np.array([len(t) for t in toks])))
g = np.array([d["gold"] for d in data]); pred = np.array([r["pred"] for r in rows])
def pick(T, mode="lse"):
    out = []
    for d in data:
        s = np.array([np.logaddexp.reduce(T * d["lg"][t]) if mode == "lse" else (T * d["lg"][t]).max() for t in d["toks"]])
        out.append(s)
    return out
base = np.array([np.argmax(s) for s in pick(1.0)])
for T in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    p = np.array([np.argmax(s) for s in pick(T)])
    print("T=%.1f acc %.3f  vs T1: w->r %d r->w %d" % (T, (p == g).mean(), ((p == g) & (base != g)).sum(), ((p != g) & (base == g)).sum()))
p = np.array([np.argmax(s) for s in pick(1.0, "max")]); print("max-token acc %.3f" % (p == g).mean())
# length bias: is the pick the longest option more than gold is?
lr = lambda arr: np.mean([np.argsort(np.argsort(-d["L"]))[x] for d, x in zip(data, arr)])
print("mean length-rank (0=longest): gold %.2f  H12 pick %.2f  final pred %.2f  T3 pick %.2f" % (lr(g), lr(base), lr(pred), lr([np.argmax(s) for s in pick(3.0)])))
print("pick==longest: gold %.3f H12 %.3f final %.3f" % tuple(np.mean([np.argmax(d["L"]) == x for d, x in zip(data, arr)]) for arr in [g, base, pred]))
# per-token mean logit vs option length (does mass grow with length?)
