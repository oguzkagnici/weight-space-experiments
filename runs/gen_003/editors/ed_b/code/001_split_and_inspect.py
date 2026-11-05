# Hypothesis: none (setup). Fix a DEV/HOLDOUT split of the practice questions and inspect prompt lengths / layer-22 attention shapes.
from wsp.editor import *
import numpy as np, json, random

Q = questions()
ids = sorted(Q.keys())
rng = random.Random(3003)
rng.shuffle(ids)
hold = ids[:768]
dev = ids[768:]
json.dump({"dev": dev, "hold": hold}, open("work/split.json", "w"))
print("n", len(ids), "dev", len(dev), "hold", len(hold))

# lengths and option counts on first 300 dev
lens, nopt = [], []
for q in dev[:300]:
    s = render(q)
    lens.append(len(tok.encode(s)))
    nopt.append(len(Q[q]["options"]))
lens = np.array(lens); nopt = np.array(nopt)
print("tokens: mean %.0f median %.0f max %d sum(200)=%d" % (lens.mean(), np.median(lens), lens.max(), lens[:200].sum()))
print("n options counts", {k: int((nopt == k).sum()) for k in sorted(set(nopt))})
print(repr(render(dev[0])[-400:]))
for n in NAMES:
    if ".layers.22." in n:
        print(n, SHAPES[n])
