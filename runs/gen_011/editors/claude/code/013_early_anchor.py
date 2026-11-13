# H13: wrong traces are ANCHORED early: at ~20% of the trace the latent answer is already the (wrong) final pick with high confidence.
import json, numpy as np
from wsp.editor import *
ix = json.load(open("work/probe1_index.json")); ids = np.load("work/fwd_01/topk_ids.npy"); lps = np.load("work/fwd_01/topk_logprobs.npy")
L = "ABCDEFGHIJ"; LI = list(LETTER_IDS)
def dist(t):
    p = np.zeros(10)
    for i, lp in zip(ids[t], lps[t]):
        if i in LI: p[LI.index(i)] += np.exp(lp)
    return p / max(p.sum(), 1e-9)
for k, name in ((1, "right"), (0, "wrong")):
    conf, eqf, ent = [], [], []
    for n, c in enumerate(ix):
        if c["correct"] != k: continue
        p = dist(3 * n); f = L.index(c["pred"])
        conf.append(p.max()); eqf.append(p.argmax() == f); ent.append(-(p * np.log(p + 1e-12)).sum())
        if k == 0 and p.argmax() == f: pass
    conf = np.array(conf); eqf = np.array(eqf)
    print(f"{name}: n {len(conf)} p0 top==final {eqf.sum()}  P(top) median {np.median(conf):.2f}  P(top)>0.9 {int((conf>0.9).sum())}  entropy median {np.median(ent):.2f}"
          f"  | when top==final: P(top) median {np.median(conf[eqf]):.2f}")
print("meta_frac (p0 position ~) median", round(float(np.median([c['meta_frac'] for c in ix])), 2))
