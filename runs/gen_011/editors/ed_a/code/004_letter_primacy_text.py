# Hypothesis: if an early-in-sweep (primacy) bias acts in thinking mode, finished answers over-pick early letters and accuracy falls with the gold letter's position.
import json, collections
import numpy as np
from wsp.editor import *
SH = "/home/user/weight-space-experiments/runs/shared/"
R = [json.loads(l) for l in open(SH + "passk_gen008champ_cap8192_det1.jsonl")]
R += [r for r in (json.loads(l) for l in open(SH + "trace_bank_gen008champ_cap8192_det1.jsonl")) if r["id"] not in {x["id"] for x in R}]
Q = questions()
fin = [r for r in R if not r["cut"] and r["pred"]]
print("finished", len(fin))
L = "ABCDEFGHIJ"
pc = collections.Counter(r["pred"] for r in fin); gc = collections.Counter(r["gold"] for r in fin)
print("pred ", [pc[c] for c in L]); print("gold ", [gc[c] for c in L])
for grp, s in [("A-C", "ABC"), ("D-G", "DEFG"), ("H-J", "HIJ")]:
    x = [r["correct"] for r in fin if r["gold"] in s]
    print(grp, "acc finished %.2f n %d" % (np.mean(x), len(x)))
# wrong answers: pred earlier or later than gold
w = [r for r in fin if not r["correct"]]
e = sum(L.index(r["pred"]) < L.index(r["gold"]) for r in w)
print("wrong: pred earlier than gold", e, "of", len(w))
# expected under uniform-wrong pick: mean over wrong of (gold index)/(n-1)
exp = sum(L.index(r["gold"]) / (len(Q[r["id"]]["options"]) - 1) for r in w)
print("expected if wrong pick uniform: %.1f" % exp)
