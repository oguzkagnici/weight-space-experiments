# Hypothesis: the long-option bias of H12 comes from summing attention over option tokens; length-normalizing the option score (lse - beta*log L) or restricting to end tokens should remove it and raise H12 gold-pick rate (lens).
import numpy as np, pickle
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
A = np.load(D + "attn_last_L22.npy")
g = np.array([o["gold"] for o in sp])
rows = pickle.load(open("work/diag03.pkl", "rb")); pred = np.array([r["pred"] for r in rows])
S, Ls, Send = [], [], []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]; att = A[a:b, 12].astype(np.float64) + 1e-9
    S.append(np.array([np.log(att[s0:s1 + 2].sum()) for (s0, s1) in o["spans"]]))
    Send.append(np.array([np.log(att[s1:s1 + 2].sum()) for (s0, s1) in o["spans"]]))
    Ls.append(np.array([s1 - s0 + 2 for (s0, s1) in o["spans"]], float))
for beta in [0, 0.25, 0.5, 1.0]:
    p = np.array([np.argmax(s - beta * np.log(L)) for s, L in zip(S, Ls)])
    print("beta %.2f acc %.3f" % (beta, (p == g).mean()))
p = np.array([np.argmax(s) for s in Send]); print("end+newline only acc %.3f" % (p == g).mean())
# where is mass: share on non-end tokens vs length
share_mid = np.mean([1 - np.exp(se - s).mean() for se, s in zip(Send, S)])
print("mean fraction of option mass not on last/newline tokens %.3f" % share_mid)
# regression of centered H12 option log-mass on centered log length
x = np.concatenate([np.log(L) - np.log(L).mean() for L in Ls]); y = np.concatenate([s - s.mean() for s in S]); ye = np.concatenate([s - s.mean() for s in Send])
print("slope logmass~logL: all tokens %.2f, end tokens %.2f" % (np.polyfit(x, y, 1)[0], np.polyfit(x, ye, 1)[0]))
gl = np.concatenate([(np.arange(len(L)) == gg) for L, gg in zip(Ls, g)])
print("slope is_gold~logL: %.3f (gold tends longer)" % np.polyfit(x, gl.astype(float), 1)[0])
