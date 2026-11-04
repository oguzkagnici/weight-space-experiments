# hypothesis: removing the final letter bias would gain several points; estimate the ceiling with per-letter constant
# offsets estimated on one half of the recorded outputs and evaluated on the other (cross-fitted). Not an edit.
import json, numpy as np
from wsp.editor import *
Q = questions()
rows = [json.loads(l) for l in open("work/fwd_01/score_1_original.jsonl")]
n = len(rows); rng = np.random.default_rng(0)
logp = [np.log(np.array(r["probs"]) + 1e-12) for r in rows]
ans = np.array([r["answer"] for r in rows]); k = np.array([len(p) for p in logp])
def acc_with(b, ids):
    return np.mean([np.argmax(logp[i] + b[:k[i]]) == ans[i] for i in ids])
res = []
for rep in range(20):
    perm = rng.permutation(n); a, c = perm[: n // 2], perm[n // 2:]
    for fit, ev in ((a, c), (c, a)):
        # offsets = minus the mean centred log-prob per letter on the fit half (10-option questions)
        m = [i for i in fit if k[i] == 10]
        cen = np.array([logp[i] - logp[i].mean() for i in m]).mean(0)
        for s in (0.5, 1.0):
            res.append((s, acc_with(-s * cen, ev) - acc_with(np.zeros(10), ev)))
for s in (0.5, 1.0):
    v = [r[1] for r in res if r[0] == s]
    print(f"offset strength {s}: cross-fitted accuracy gain {100*np.mean(v):+.2f} points (spread {100*np.std(v):.2f})")
