# Hypothesis: same paired evaluation as 018 for the DEV passk questions (keyed by id+sample) and the 3,072 holdout bank.
import json, sys, glob
import numpy as np
bankf, dirs = sys.argv[1], sys.argv[2:]
key = lambda r: (r["id"], r.get("sample", 0))
bank = {key(json.loads(l)): json.loads(l) for l in open(bankf)}
rows = {}
for d in dirs:
    for f in glob.glob(d + "/gen_*_edited.jsonl"):
        for l in open(f):
            r = json.loads(l); rows[key(r)] = r
ids = [i for i in rows if i in bank]
def paired(sel):
    d = np.array([rows[i]["correct"] - bank[i]["correct"] for i in sel])
    return 100 * d.mean(), 100 * d.std(ddof=1) / np.sqrt(len(d)), int((d > 0).sum()), int((d < 0).sum()), len(d)
print("overall: parent %.1f -> edited %.1f ; diff %+.2f (SE %.2f) +%d/-%d n %d" % ((np.mean([bank[i]["correct"] for i in ids]) * 100, np.mean([rows[i]["correct"] for i in ids]) * 100) + paired(ids)))
bf = [i for i in ids if not bank[i]["cut"] and not rows[i]["cut"]]
print("both-finish: diff %+.2f (SE %.2f) +%d/-%d n %d" % paired(bf))
print("cut parent %d edited %d ; identical %d" % (sum(bank[i]["cut"] for i in ids), sum(rows[i]["cut"] for i in ids), sum(bank[i]["raw"] == rows[i]["raw"] for i in ids)))
byq = {}
for i in ids: byq.setdefault(i[0], []).append((bank[i]["pred"], rows[i]["pred"], bank[i]["gold"]))
for q, v in byq.items():
    if len(v) > 1: print(q, "gold", v[0][2], "parent", "".join(x[0] or "-" for x in v), "edited", "".join(x[1] or "-" for x in v))
