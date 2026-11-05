# Hypothesis check: which L22 heads spend attention on option letter tokens, and does dropping those tokens from each head's option vote change its gold-pick (lens, DEV[:240])?
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb")); off = np.load(D + "offsets_tok.npy")
A = np.load(D + "attn_last_L22.npy").astype(np.float64)
for h in range(16):
    lm, om, a0, a1 = [], [], [], []
    for i, o in enumerate(sp):
        att = A[off[i]:off[i + 1], h] + 1e-9
        lm.append(sum(att[s0] for s0, s1 in o["spans"])); om.append(sum(att[s0:s1 + 2].sum() for s0, s1 in o["spans"]))
        m0 = [att[s0:s1 + 2].sum() for s0, s1 in o["spans"]]; m1 = [att[s0 + 1:s1 + 2].sum() for s0, s1 in o["spans"]]
        a0.append(np.argmax(m0) == o["gold"]); a1.append(np.argmax(m1) == o["gold"])
    a0 = np.array(a0); a1 = np.array(a1)
    print("H%-2d letter-token mass %.3f option mass %.3f | pick acc %.3f -> drop letters %.3f (+%d/-%d)" % (h, np.mean(lm), np.mean(om), a0.mean(), a1.mean(), (a1 & ~a0).sum(), (~a1 & a0).sum()))
