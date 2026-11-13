# H7: content-free option-feature biases in the thinking model's picks: option LENGTH (prefers longer/more specific?) and hedge options ("both", "all of the above", "none", "depending").
import json, re, numpy as np, collections
from wsp.editor import questions
Q = questions()
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
for p in ["trace_bank_gen008champ_cap8192_det1.jsonl", "trace_bank_gen008champ_cap3072_det1.jsonl"]:
    rows += [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/" + p)]
L = "ABCDEFGHIJ"
fin = [r for r in rows if not r["cut"] and r.get("pred") in L]
print("finished answers", len(fin), "questions", len({r["id"] for r in fin}))
# length rank (0 = shortest .. 1 = longest) of pred vs gold, among wrong answers
pr, gr = [], []
for r in fin:
    q = Q[r["id"]]; lens = np.array([len(o) for o in q["options"]]); n = len(lens)
    rank = lambda i: (lens < lens[i]).sum() / (n - 1)
    pi = L.index(r["pred"]); gi = q["answer_index"]
    if not r["correct"]: pr.append(rank(pi)); gr.append(rank(gi))
print("wrong answers: mean length rank pred", round(np.mean(pr), 3), "gold", round(np.mean(gr), 3), "n", len(pr),
      "pred longer than gold", sum(p > g for p, g in zip(pr, gr)), "shorter", sum(p < g for p, g in zip(pr, gr)))
# overall: length rank of gold over all practice questions, and of picks
allg = []
for q in Q.values():
    lens = np.array([len(o) for o in q["options"]]); n = len(lens)
    if n > 1: allg.append((lens < lens[q["answer_index"]]).sum() / (n - 1))
print("practice: mean length rank of gold", round(np.mean(allg), 3), "share gold is longest", round(np.mean([g == 1 for g in allg]), 3))
hed = re.compile(r"\b(both|all of the above|none of the above|depend|all of these|neither)\b", re.I)
c = collections.Counter()
for r in fin:
    q = Q[r["id"]]; hs = [i for i, o in enumerate(q["options"]) if hed.search(o)]
    if not hs: continue
    c["q_with_hedge_opt"] += 1; c["gold_is_hedge"] += q["answer_index"] in hs; c["pred_is_hedge"] += L.index(r["pred"]) in hs
print(dict(c))
gh = sum(1 for q in Q.values() if any(hed.search(o) for o in q["options"]))
gg = sum(1 for q in Q.values() if hed.search(q["options"][q["answer_index"]]))
print("practice: questions with hedge option", gh, "gold is hedge option", gg)
