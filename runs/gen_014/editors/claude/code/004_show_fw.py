# Hypothesis: (inspection) print stems, gold and wrong picks of the finished-wrong verbal questions to hand-label the decisive stem cue.
import json, sys, collections
from wsp.editor import *
Q = questions()
QUANT = {"business","chemistry","physics","math","engineering"}
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pkids={r["id"] for r in pk}
allr = pk + [dict(r, sample=0) for r in bk if r["id"] not in pkids]
by = collections.defaultdict(list)
for r in allr: by[r["id"]].append(r)
done = set(sys.argv[1].split(","))
for q in sorted(by):
    d = Q[q]
    if d["category"] in QUANT or q in done: continue
    if not any((not r["cut"]) and not r["correct"] for r in by[q]): continue
    print("=====", q, d["category"], "gold", d["answer"], "picks", [(r["pred"] if not r["cut"] else "cut") for r in sorted(by[q], key=lambda r:r["sample"])])
    print(d["question"][:700].encode("ascii","replace").decode())
    for i,o in enumerate(d["options"]): print(" ", "ABCDEFGHIJ"[i], o[:80].encode("ascii","replace").decode())
