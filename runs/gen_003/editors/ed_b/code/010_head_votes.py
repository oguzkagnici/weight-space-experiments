# Hypothesis check: do other attention heads (L14/L20/L22) select options with evidence independent of L22.H12, such that a combination of their option-attention votes beats H12 alone? (Lens-level ceiling for any 'ensemble' edit.)
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
g = np.array([o["gold"] for o in sp])
votes = {}
for L in [14, 20, 22]:
    A = np.load(D + "attn_last_L%d.npy" % L)
    for h in range(16):
        S = []
        for i, o in enumerate(sp):
            a, b = off[i], off[i + 1]; att = A[a:b, h].astype(np.float64)
            m = np.array([att[s0:s1 + 2].sum() for (s0, s1) in o["spans"]])
            S.append(np.log(m + 1e-6))
        votes[(L, h)] = S
res = []
for k, S in votes.items():
    acc = np.mean([np.argmax(s) == gg for s, gg in zip(S, g)])
    om = np.mean([np.exp(s).sum() for s in S])
    res.append((acc, k, om))
res.sort(reverse=True)
print("top heads by option-mass argmax acc (optmass):", [(f"L{k[0]}H{k[1]}", round(a, 3), round(om, 2)) for a, k, om in res[:10]])
base = votes[(22, 12)]
b0 = np.array([np.argmax(s) == gg for s, gg in zip(base, g)])
for acc, k, om in res[1:8]:
    for w in [0.25, 0.5, 1.0]:
        c = np.array([np.argmax(s1 + w * s2) == gg for s1, s2, gg in zip(base, votes[k], g)])
        print(f"H12 + {w}*L{k[0]}H{k[1]}: acc {c.mean():.3f} (w->r {(c & ~b0).sum()} r->w {(~c & b0).sum()})")
