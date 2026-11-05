# Hypothesis check: L22 heads 14 and 15 select options about as well as H12 (acc .40); are their picks independent enough that a combined vote beats H12 (ceiling for converting a suppression head into a copy head)?
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb")); off = np.load(D + "offsets_tok.npy")
A = np.load(D + "attn_last_L22.npy"); g = np.array([o["gold"] for o in sp])
S = {}
for h in range(16):
    S[h] = []
    for i, o in enumerate(sp):
        a, b = off[i], off[i + 1]; att = A[a:b, h].astype(np.float64)
        S[h].append(np.log(np.array([att[s0:s1 + 2].sum() for (s0, s1) in o["spans"]]) + 1e-6))
pk = {h: np.array([np.argmax(s) for s in S[h]]) for h in S}
for h in [11, 13, 14, 15, 10, 0]:
    print("H%d acc %.3f agree with H12 %.3f; right when H12 wrong %d/%d" % (h, (pk[h] == g).mean(), (pk[h] == pk[12]).mean(), ((pk[h] == g) & (pk[12] != g)).sum(), (pk[12] != g).sum()))
b0 = pk[12] == g
for combo in [(12, 14), (12, 15), (12, 14, 15), (12, 13, 14, 15), (12, 11, 14, 15)]:
    for w in [0.5, 1.0]:
        c = np.array([np.argmax(S[12][i] + w * sum(S[h][i] for h in combo[1:])) for i in range(len(g))]) == g
        print(combo, w, "acc %.3f w->r %d r->w %d" % (c.mean(), (c & ~b0).sum(), (~c & b0).sum()))
