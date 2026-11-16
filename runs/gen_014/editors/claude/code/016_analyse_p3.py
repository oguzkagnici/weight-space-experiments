# Hypothesis: the L22.H10 x1.5 edit raises P(gold) at the probe on the pass-1 texts (paired with pass 1) and enlarges the cue effect; expected ~null.
import json, numpy as np, collections
from wsp.editor import *
meta = json.load(open("work/p1_meta.json")); LET = "ABCDEFGHIJ"
MIX = {"mmlupro_1171","mmlupro_2654","mmlupro_6110","mmlupro_6562","mmlupro_6800"}
def load(k):
    tk = np.load(f"work/fwd_{k}/topk_ids.npy"); tl = np.load(f"work/fwd_{k}/topk_logprobs.npy"); out = []
    for i, m in enumerate(meta):
        d = {}
        for L, t in zip(LET, LETTER_IDS):
            w = np.where(tk[i] == t)[0]; d[L] = tl[i][w[0]] if len(w) else -30.0
        tot = np.logaddexp.reduce(list(d.values())); out.append(dict(pg=np.exp(d[m["gold"]] - tot), top=max(d, key=d.get), mass=np.exp(tot)))
    return out
O, E = load("01"), load("03")
for sub, f in (("all", lambda m: True), ("not used to pick H10", lambda m: m["id"] not in MIX)):
    for cut in ("early", "late"):
        for v in ("none", "ctrl", "cue"):
            ix = [i for i, m in enumerate(meta) if m["cut"] == cut and m["variant"] == v and f(m)]
            if not ix: continue
            qm = collections.defaultdict(list)
            for i in ix: qm[meta[i]["id"]].append(E[i]["pg"] - O[i]["pg"])
            qv = np.array([np.mean(x) for x in qm.values()])
            se = qv.std(ddof=1) / np.sqrt(len(qv)) if len(qv) > 1 else float("nan")
            print(f"{sub:22s} {cut:5s} {v:4s} n {len(ix):2d}: dP(gold) edited-original {qv.mean():+.4f} (SE {se:.4f}); top=gold {sum(O[i]['top']==meta[i]['gold'] for i in ix)} -> {sum(E[i]['top']==meta[i]['gold'] for i in ix)}; letter mass {np.mean([O[i]['mass'] for i in ix]):.2f} -> {np.mean([E[i]['mass'] for i in ix]):.2f}")
