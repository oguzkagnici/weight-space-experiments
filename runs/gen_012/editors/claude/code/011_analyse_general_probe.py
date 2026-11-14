# Analyse pass 2: (a) H2b letter lens per layer at wrong-trace probes (50% / end) vs right controls; (b) H9 prior-probe letter statistics,
# cross-fitted per-letter offsets c_L = log(pred share) - log(gold share) (fit on half A, evaluated on half B and vice versa).
import json, numpy as np
from wsp.editor import *
D = "work/fwd_02/"; ix = json.load(open("work/probe2_index.json")); n = len(ix)
LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"
E = load("model.embed_tokens.weight")[LI].numpy(); gf = load("model.embedding_norm.weight").numpy()
rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
def outp(t):
    p = np.zeros(10)
    for a, b in zip(tk[t], tl[t]):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    return p
P = np.array([outp(t) for t in range(n)]); mass = P.sum(1); P = P / np.maximum(mass[:, None], 1e-9)
gold = np.array([LET.index(i["gold"]) for i in ix]); kind = np.array([i["kind"] for i in ix])
g = kind == "gen"; corr = np.array([i.get("correct", -1) for i in ix]); frac = np.array([i.get("frac", 0) for i in ix])
print("letter mass in top-20: gen probes", mass[g].mean().round(3), "prior probes", mass[~g].mean().round(3))
print("(a) lens argmax==gold | P10(gold) by layer")
groups = [("wrong 50%", g & (corr == 0) & (frac == 0.5)), ("wrong end", g & (corr == 0) & (frac == 1.0)), ("right 50%", g & (corr == 1))]
print("layer " + "".join(f"{lab:>20s}" for lab, _ in groups))
for l in [14, 15, 16, 18, 20, 22, 23, 24]:
    h = np.load(D + f"h{l}.npy"); lg = (h / rms(h) * gf) @ E.T; z = np.exp(lg - lg.max(1, keepdims=True)); p = z / z.sum(1, keepdims=True)
    print(f"h{l:2d}  " + "".join(f"{np.mean(lg[m].argmax(1) == gold[m]):>12.2f} {p[m, gold[m]].mean():.2f}  " for _, m in groups))
m = g & (corr == 0) & (P.argmax(1) != gold)
cnt = 0
for l in [14, 15, 16, 18, 20, 22, 23]:
    h = np.load(D + f"h{l}.npy"); lg = (h / rms(h) * gf) @ E.T; cnt += np.sum(lg[m].argmax(1) == gold[m])
    print(f"  wrong probes (output != gold, n {m.sum()}): gold = lens top at h{l}: {np.sum(lg[m].argmax(1) == gold[m])}")
# (b) prior statistics
pr = ~g; Pp = P[pr]; gp_ = gold[pr]
print("\n(b) prior probes", pr.sum(), "acc", np.mean(Pp.argmax(1) == gp_).round(3), "mean P(gold)", Pp[np.arange(len(gp_)), gp_].mean().round(3),
      "mean log P(gold)", np.log(Pp[np.arange(len(gp_)), gp_] + 1e-9).mean().round(3))
print("pred counts", np.bincount(Pp.argmax(1), minlength=10).tolist()); print("gold counts", np.bincount(gp_, minlength=10).tolist())
print("mean P per letter", Pp.mean(0).round(3).tolist())
rng = np.random.default_rng(0); idx = rng.permutation(len(gp_)); halves = [idx[: len(idx) // 2], idx[len(idx) // 2:]]
res = []
for a, b in [(0, 1), (1, 0)]:
    A, B = halves[a], halves[b]
    share_p = Pp[A].mean(0); share_g = (np.bincount(gp_[A], minlength=10) + 1) / (len(A) + 10)
    c = np.log(share_p) - np.log(share_g); c -= c.mean()
    for s in (0.5, 1.0):
        lq = np.log(Pp[B] + 1e-9) - s * c; q = np.exp(lq - lq.max(1, keepdims=True)); q /= q.sum(1, keepdims=True)
        res.append((a, s, np.mean(q.argmax(1) == gp_[B]) - np.mean(Pp[B].argmax(1) == gp_[B]), (np.log(q[np.arange(len(B)), gp_[B]]) - np.log(Pp[B][np.arange(len(B)), gp_[B]] + 1e-9)).mean()))
    print(f"fit half {a}: c =", np.round(c, 2).tolist())
for r in res: print(f"  fit {r[0]} strength {r[1]}: held-out acc change {100 * r[2]:+.1f} pts, mean dlogP(gold) {r[3]:+.3f}")
json.dump({"P_prior_mean": Pp.mean(0).tolist(), "gold_counts": np.bincount(gp_, minlength=10).tolist()}, open("work/prior_stats.json", "w"))
