# Hypothesis (gate, step 2): quoting the decisive cue raises log P(gold) - log P(wrong pick) at the forced conclusion, vs quoting a non-decisive stem sentence.
import json, numpy as np, collections
from wsp.editor import *
D = "work/fwd_01/"; meta = json.load(open("work/p1_meta.json"))
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
print("topk shapes", tk.shape, tl.shape)
LET = "ABCDEFGHIJ"
def letdist(i):
    ids = tk[i].reshape(-1); lp = tl[i].reshape(-1); d = {}
    for L, t in zip(LET, LETTER_IDS):
        w = np.where(ids == t)[0]; d[L] = lp[w[0]] if len(w) else -30.0
    return d
rec = {}
for i, m in enumerate(meta):
    d = letdist(i); g, p = m["gold"], m["pred"]
    tot = np.logaddexp.reduce(list(d.values()))
    rec[(m["id"], m["sample"], m["cut"], m["variant"])] = dict(margin=d[g] - d[p], pg=np.exp(d[g] - tot), pp=np.exp(d[p] - tot), mass=np.exp(tot), top=max(d, key=d.get), g=g, p=p)
for cut in ("early", "late"):
    for a, b in (("cue", "ctrl"), ("cue", "none"), ("ctrl", "none")):
        diffs, keys = [], []
        for (q, s, c, v), r in rec.items():
            if c != cut or v != a or (q, s, c, b) not in rec: continue
            diffs.append(r["margin"] - rec[(q, s, c, b)]["margin"]); keys.append(q)
        if not diffs: continue
        diffs = np.array(diffs)
        # cluster by question
        qm = collections.defaultdict(list)
        for k, x in zip(keys, diffs): qm[k].append(x)
        qv = np.array([np.mean(v) for v in qm.values()])
        print(f"{cut:5s} {a}-{b}: n {len(diffs)} samples / {len(qv)} q; margin change mean {diffs.mean():+.3f} (SE by question {qv.std(ddof=1)/np.sqrt(len(qv)):.3f}); "
              f"up in {(diffs > 0).sum()}/{len(diffs)}; median {np.median(diffs):+.3f}")
for cut in ("early", "late"):
    for v in ("none", "ctrl", "cue"):
        rs = [r for (q, s, c, vv), r in rec.items() if c == cut and vv == v]
        if rs: print(cut, v, "n", len(rs), "P(gold) %.3f P(pick) %.3f letter mass %.2f top=gold %d top=pick %d" % (np.mean([r["pg"] for r in rs]), np.mean([r["pp"] for r in rs]), np.mean([r["mass"] for r in rs]), sum(r["top"] == r["g"] for r in rs), sum(r["top"] == r["p"] for r in rs)))
print("\nper question (late cue - ctrl margin; early cue - ctrl mean):")
for q in sorted({k[0] for k in rec}):
    late = [rec[k]["margin"] - rec[(k[0], k[1], k[2], "ctrl")]["margin"] for k in rec if k[0] == q and k[2] == "late" and k[3] == "cue"]
    early = [rec[k]["margin"] - rec[(k[0], k[1], k[2], "ctrl")]["margin"] for k in rec if k[0] == q and k[2] == "early" and k[3] == "cue"]
    print(q, "late", np.round(late, 2), "early", np.round(early, 2))
