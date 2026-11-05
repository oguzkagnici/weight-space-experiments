# Hypothesis: L22.H12's attention (pure q.k, NoPE) picks gold more often than the final answer; the gap comes from soft attention / letter priors. Check accuracy of final vs H12 argmax, and where in key space the correct-option signal lives.
import numpy as np, pickle, json
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
tki = np.load(D + "topk_ids.npy"); tkl = np.load(D + "topk_logprobs.npy")
A = np.load(D + "attn_last_L22.npy")
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
LET = list(range(41, 51))
H = 12; G = H // 4
rows = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    assert b - a == o["T"]
    last = b - 1
    lp = np.full(o["n"], -30.0)
    for j in range(o["n"]):
        w = np.where(tki[last] == LET[j])[0]
        if len(w): lp[j] = tkl[last, w[0]]
    pred = int(np.argmax(lp))
    att = A[a:b, H]
    mass = np.array([att[s0:s1 + 2].sum() for (s0, s1) in o["spans"]])  # include trailing newline
    endatt = np.array([att[s1] for (s0, s1) in o["spans"]])
    tokarg = int(np.argmax(att))
    tokopt = -1
    for j, (s0, s1) in enumerate(o["spans"]):
        if s0 <= tokarg <= s1 + 1: tokopt = j
    # attention logits check
    q = np.asarray(Qn[last, H]); K = np.asarray(Kn[a:b, G])
    lg = K @ q / np.sqrt(128)
    p = np.exp(lg - lg.max()); p /= p.sum()
    rows.append(dict(gold=o["gold"], pred=pred, n=o["n"], massarg=int(np.argmax(mass)), tokopt=tokopt,
                     endarg=int(np.argmax(endatt)), optmass=mass.sum(), err=np.abs(p - att).max(),
                     lpg=lp[o["gold"]] - np.log(np.exp(lp).sum())))
g = np.array([r["gold"] for r in rows])
for k in ["pred", "massarg", "tokopt", "endarg"]:
    v = np.array([r[k] for r in rows]); print(k, "acc %.3f" % (v == g).mean())
pr = np.array([r["pred"] for r in rows]); ma = np.array([r["massarg"] for r in rows])
print("agree pred/massarg %.3f" % (pr == ma).mean())
print("massarg right & pred wrong:", int(((ma == g) & (pr != g)).sum()), " pred right & massarg wrong:", int(((ma != g) & (pr == g)).sum()))
print("mean option mass %.3f, max |p_qk - att| %.4f" % (np.mean([r["optmass"] for r in rows]), max(r["err"] for r in rows)))
print("mean logp gold %.3f" % np.mean([r["lpg"] for r in rows]))
ten = np.array([r["n"] == 10 for r in rows])
print("10-opt: pred counts", np.bincount(pr[ten], minlength=10), "massarg", np.bincount(ma[ten], minlength=10), "gold", np.bincount(g[ten], minlength=10))
pickle.dump(rows, open("work/diag03.pkl", "wb"))
