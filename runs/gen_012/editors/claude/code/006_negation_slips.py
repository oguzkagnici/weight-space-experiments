# H5 (free): is a lost negation ("NOT", "EXCEPT", "least", "incorrect", "false") a recurring reasoning slip? Compare accuracy of negation
# questions vs others in the 8,192 bank + passk, and how many practice questions have negations (coverage).
import json, re, numpy as np
from wsp.editor import *
Q = questions()
NEG = re.compile(r"\b(NOT|EXCEPT|LEAST|not true|incorrect|false|least likely|would not|does not|is not|cannot|isn't|except)\b")
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
rows += [dict(json.loads(l), sample=0) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
seen = set(); R = []
for r in rows:
    k = (r["id"], r.get("sample", 0))
    if k in seen: continue
    seen.add(k); R.append(r)
neg = lambda q: bool(NEG.search(Q[q]["question"]))
for lab, f in [("negation", True), ("other", False)]:
    rr = [r for r in R if neg(r["id"]) == f]; fin = [r for r in rr if not r["cut"]]
    print(f"{lab:9s} answers {len(rr)} questions {len({r['id'] for r in rr})} acc {np.mean([r['correct'] for r in rr]):.3f} finished acc {np.mean([r['correct'] for r in fin]):.3f}")
allq = list(Q); print("practice share with negation", round(np.mean([neg(q) for q in allq]), 3))
caps = re.compile(r"\b(NOT|EXCEPT|LEAST)\b")
rr = [r for r in R if caps.search(Q[r["id"]]["question"])]
print("capitalised NOT/EXCEPT/LEAST answers", len(rr), "acc", np.mean([r["correct"] for r in rr]).round(3) if rr else None, "share practice", round(np.mean([bool(caps.search(Q[q]['question'])) for q in allq]), 3))
