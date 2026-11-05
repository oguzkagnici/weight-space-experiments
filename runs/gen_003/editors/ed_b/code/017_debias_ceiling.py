# Hypothesis: (ceiling) a perfect constant per-letter offset at the final logits, cross-fitted, would gain how much on DEV2 (512)? Separately for 10-option and fewer-option questions.
import numpy as np, json
rows = [json.loads(l) for l in open("work/fwd_02/score_2_original.jsonl")]
n = len(rows); g = np.array([r["answer"] for r in rows])
LP = [np.log(np.array(r["probs"]) + 1e-9) for r in rows]
ten = np.array([len(p) == 10 for p in LP])
idx = np.where(ten)[0]
rng = np.random.default_rng(0)
gains = []
for rep in range(20):
    perm = rng.permutation(idx); folds = [perm[::2], perm[1::2]]
    tot0 = tot1 = 0
    for f in range(2):
        tr, te = folds[1 - f], folds[f]
        C = np.array([LP[i] - LP[i].mean() for i in tr])
        b = C.mean(0)
        for i in te:
            tot0 += np.argmax(LP[i]) == g[i]; tot1 += np.argmax(LP[i] - b) == g[i]
    gains.append((tot1 - tot0) / len(idx))
print("10-opt n=%d acc %.3f; CV constant-debias gain mean %.3f (sd over splits %.3f)" % (len(idx), np.mean([np.argmax(LP[i]) == g[i] for i in idx]), np.mean(gains), np.std(gains)))
# fraction of bias -> also logp gain
