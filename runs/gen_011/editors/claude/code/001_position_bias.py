# H1: early option judgement has a content-free position bias (satisficing): wrong picks are earlier letters than gold more often than chance, and accuracy falls with gold position.
import json, numpy as np, collections
from wsp.editor import questions
Q = questions()
rows = []
for p in ["/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl",
          "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl",
          "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"]:
    for l in open(p):
        r = json.loads(l); r["src"] = p.split("/")[-1][:12]; rows.append(r)
# dedupe trace bank 8192 sample0 vs passk sample0
seen = set(); R = []
for r in rows:
    k = (r["id"], r.get("sample", 0), "3072" in r["src"] or "cap3072" in r["src"])
    if k in seen: continue
    seen.add(k); R.append(r)
print(len(R), collections.Counter(r["src"] for r in R))
L = "ABCDEFGHIJ"
fin = [r for r in R if not r["cut"] and r.get("pred")]
print("finished", len(fin))
# accuracy by gold index (finished)
by = collections.defaultdict(list)
for r in fin:
    q = Q[r["id"]]; by[q["answer_index"]].append(r["correct"])
print("gold idx: n acc", [(i, len(by[i]), round(100*np.mean(by[i]))) for i in sorted(by)])
# wrong picks: pred before gold?
earlier = later = 0; exp = []
for r in fin:
    if r["correct"]: continue
    q = Q[r["id"]]; g = q["answer_index"]; n = len(q["options"]); p = L.index(r["pred"]) if r["pred"] in L else None
    if p is None: continue
    if p < g: earlier += 1
    else: later += 1
    exp.append(g / (n - 1))
print("wrong picks: earlier than gold", earlier, "later", later, "expected earlier share under uniform wrong pick", round(np.mean(exp), 3))
# pred index distribution vs gold index distribution
pc = collections.Counter(L.index(r["pred"]) for r in fin if r["pred"] in L)
gc = collections.Counter(Q[r["id"]]["answer_index"] for r in fin)
print("pred idx", [pc[i] for i in range(10)]); print("gold idx", [gc[i] for i in range(10)])
