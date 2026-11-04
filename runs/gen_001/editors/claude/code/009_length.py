# hypothesis H3: accuracy collapses on long prompts (options far from the answer position / beyond the 256 sliding
# window), independent of subject.
import json, numpy as np
from wsp.editor import *
rows = [json.loads(l) for l in open("work/fwd_01/score_1_original.jsonl")]
Q = questions()
L = np.array([r["n_tokens"] for r in rows]); c = np.array([r["correct"] for r in rows]); cat = [r["category"] for r in rows]
for lo, hi in ((0, 120), (120, 180), (180, 256), (256, 400), (400, 5000)):
    m = (L >= lo) & (L < hi)
    print(f"tokens {lo:4d}-{hi:4d}: n {m.sum():3d} acc {c[m].mean() if m.sum() else float('nan'):.3f}")
# within-subject: correlation of length rank and correctness
import collections
d = []
for k in set(cat):
    idx = [i for i in range(len(rows)) if cat[i] == k]
    med = np.median(L[idx]); d.append((k, c[[i for i in idx if L[i] <= med]].mean(), c[[i for i in idx if L[i] > med]].mean(), int(med)))
print("within subject: short half acc vs long half acc (median tokens)")
for x in sorted(d): print(f"  {x[0]:18s} {x[1]:.2f} vs {x[2]:.2f} (median {x[3]})")
print("overall within-subject diff (long - short):", np.mean([x[2] - x[1] for x in d]).round(3))
