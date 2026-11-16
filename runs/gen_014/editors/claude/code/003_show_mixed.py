# Hypothesis: (inspection) the decisive cue of each mixed verbal question can be identified from its stem; print stems/options/gold and sample picks.
import json, sys, collections
from wsp.editor import *
Q = questions()
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
by = collections.defaultdict(list)
for r in pk: by[r["id"]].append(r)
ids = sys.argv[1].split(",")
for q in ids:
    d = Q[q]
    print("=====", q, d["category"], "gold", d["answer"])
    print(d["question"][:900].encode("ascii","replace").decode())
    for i,o in enumerate(d["options"]): print(" ", "ABCDEFGHIJ"[i], o[:110].encode("ascii","replace").decode())
    print("  samples:", [(r["sample"], r["pred"], "cut" if r["cut"] else "", r["tokens"]) for r in sorted(by[q], key=lambda r:r["sample"])])
