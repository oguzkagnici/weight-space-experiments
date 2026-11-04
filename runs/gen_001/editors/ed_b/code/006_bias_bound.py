# Hypothesis: a content-independent per-letter offset (early-letter bias) costs accuracy; estimate the gain from removing it (fit offset on one half of DEV-512, test on the other) to decide if the bias is worth localizing.
import json, numpy as np
from wsp.editor import *
rows = [json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q = questions(); n = len(rows)
nopt = np.array([len(Q[r['id']]['options']) for r in rows]); ans = np.array([r['answer'] for r in rows])
mask = np.arange(10)[None] < nopt[:, None]
LL = np.load('work/lens_p1.npy')
def ev(lg, idx):
    lg = np.where(mask, lg, -1e9)[idx]; a = ans[idx]
    lse = np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1)) + lg.max(1)
    return round((lg.argmax(1) == a).mean(), 4), round(float((lg[np.arange(len(a)), a] - lse).mean()), 3)
rng = np.random.default_rng(0)
for layer in [24]:
    L = LL[layer]
    for rep in range(3):
        perm = rng.permutation(n); h1, h2 = perm[:n // 2], perm[n // 2:]
        m10 = nopt == 10
        fit = h1[m10[h1]]
        Lc = L - L.mean(1, keepdims=True)
        off = Lc[fit].mean(0)
        # gold-frequency correction: offset relative to log gold freq is unknown; just remove mean
        for a in [0.5, 1.0]:
            print(rep, a, 'test base', ev(L, h2), 'debiased', ev(L - a * off, h2), 'offset', np.round(off, 2))
