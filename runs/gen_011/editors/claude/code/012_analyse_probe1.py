# Analyse pass 1: change of P(gold) and P(final pick) across the ordinary paragraph (p0->p1) vs the exam-meta paragraph (p1->p2).
import json, numpy as np, re, sys
from wsp.editor import *
k = sys.argv[1] if len(sys.argv) > 1 else "01"
ix = json.load(open("work/probe1_index.json")); ids = np.load(f"work/fwd_{k}/topk_ids.npy"); lps = np.load(f"work/fwd_{k}/topk_logprobs.npy")
L = "ABCDEFGHIJ"; LI = list(LETTER_IDS)
def dist(t):
    p = np.zeros(10)
    for i, lp in zip(ids[t], lps[t]):
        if i in LI: p[LI.index(i)] += np.exp(lp)
    return p
rows = []
for n, c in enumerate(ix):
    P = [dist(3 * n + j) for j in range(3)]
    g = L.index(c["gold"]); f = L.index(c["pred"])
    mass = [p.sum() for p in P]; Pn = [p / max(p.sum(), 1e-9) for p in P]
    rows.append(dict(c=c["correct"], mass=min(mass), g=[p[g] for p in Pn], f=[p[f] for p in Pn], top=[int(p.argmax()) for p in Pn], gi=g, fi=f,
                     mcq=bool(re.search(r"multiple[- ]choice", c["para"], re.I))))
print("letter mass min median", round(float(np.median([r["mass"] for r in rows])), 3))
for k2, name in ((1, "right"), (0, "wrong"), (None, "all")):
    R = [r for r in rows if k2 is None or r["c"] == k2]
    g = np.array([r["g"] for r in R]); f = np.array([r["f"] for r in R])
    dg_ord = g[:, 1] - g[:, 0]; dg_meta = g[:, 2] - g[:, 1]; df_ord = f[:, 1] - f[:, 0]; df_meta = f[:, 2] - f[:, 1]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f"{name:5s} n {len(R)}  P(gold) p0 {g[:,0].mean():.3f} p1 {g[:,1].mean():.3f} p2 {g[:,2].mean():.3f} | dP(gold) ordinary {dg_ord.mean():+.3f} ({se(dg_ord):.3f}) meta {dg_meta.mean():+.3f} ({se(dg_meta):.3f})"
          f" | dP(final) ordinary {df_ord.mean():+.3f} meta {df_meta.mean():+.3f}")
    tg = np.array([[t == r["gi"] for t in r["top"]] for r in R]); print("       top==gold at p0/p1/p2", tg.sum(0).tolist(), " top==final", np.array([[t == r["fi"] for t in r["top"]] for r in R]).sum(0).tolist())
R = [r for r in rows if r["mcq"]]; g = np.array([r["g"] for r in R]) if R else None
if R: print("paragraphs saying 'multiple choice':", len(R), "dP(gold) ordinary", round(float((g[:,1]-g[:,0]).mean()),3), "meta", round(float((g[:,2]-g[:,1]).mean()),3))
