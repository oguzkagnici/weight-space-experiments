# H17: the no-thinking prior persists in the latent answer during early thinking (carrier exists) and samples that keep it end wrong.
import json, numpy as np, collections
from wsp.editor import *
ix = json.load(open("work/probe2_index.json")); tr = json.load(open("work/probe2_traj.json")); Q = questions()
meta = json.load(open("work/fwd_02/meta.json")); I = np.load("work/fwd_02/topk_ids.npy", mmap_mode="r"); Lg = np.load("work/fwd_02/topk_logprobs.npy", mmap_mode="r")
o = np.cumsum([0] + meta["n_tokens"]); LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"; CK = [25, 50, 100, 200, 400, 800]
prior = {}
for t, m in enumerate(ix):
    if m["sample"] is None:
        p = np.zeros(10); pos = o[t + 1] - 1
        for i, lp in zip(I[pos], Lg[pos]):
            if int(i) in LI: p[LI.index(int(i))] += np.exp(lp)
        prior[m["id"]] = p / p.sum()
mm = {(m["id"], m["sample"]): m for m in ix if m["sample"] is not None}
tab = {0: np.zeros(len(CK)), 1: np.zeros(len(CK))}; cnt = collections.Counter(); pp = {0: [], 1: []}
for key, d in tr.items():
    q, s = key.split("|"); m = mm[(q, int(s))]; c = m["correct"]; cnt[c] += 1
    pr = int(prior[q].argmax())
    tab[c] += np.array([int(np.argmax(x)) == pr for x in d]); pp[c].append([x[pr] for x in d])
for c, name in ((1, "right"), (0, "wrong")):
    print(f"{name} n {cnt[c]}: latent==prior at ck {CK}: {tab[c].astype(int).tolist()}  mean P(prior letter): {np.mean(pp[c],0).round(2).tolist()}")
print("prior confidence (max P) per question:", sorted([round(float(p.max()), 2) for p in prior.values()]))
print("prior==gold", sum(LET[p.argmax()] == Q[q]["answer"] for q, p in prior.items()), "/", len(prior))
