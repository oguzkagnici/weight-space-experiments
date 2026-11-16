# Hypothesis: the pass@4 bank at 8,192 has enough mixed (right+wrong finished) verbal questions to compare right vs wrong samples within question.
import json, collections
from wsp.editor import *
Q = questions()
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
by = collections.defaultdict(list)
for r in rows: by[r["id"]].append(r)
QUANT = {"business","chemistry","physics","math","engineering"}
mixed = []
for qid, rs in by.items():
    fin = [r for r in rs if not r["cut"]]
    nr = sum(r["correct"] for r in fin); nw = sum(1 for r in fin if not r["correct"])
    cat = Q[qid]["category"]
    if nr and nw: mixed.append((qid, cat, nr, nw, sum(r["cut"] for r in rs)))
print("questions", len(by), "rows", len(rows))
print("mixed (>=1 right & >=1 finished-wrong):", len(mixed), " verbal:", sum(1 for m in mixed if m[1] not in QUANT))
for m in mixed: print(m)
cats = collections.Counter(Q[q]["category"] for q in by)
print(cats)
