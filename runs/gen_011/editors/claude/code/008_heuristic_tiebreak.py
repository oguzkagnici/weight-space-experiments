# H8: exam-style tie-break heuristics ("more precise/specific", "most straightforward", "the question likely expects", "in multiple choice") near the final decision mark wrong conclusions.
import json, re, numpy as np, collections
from wsp.editor import questions
Q = questions()
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pk = {(r["id"], r["sample"]) for r in rows}; rows += [r for r in bank if (r["id"], 0) not in pk]
H = {"spec": r"more (precise|specific|accurate|comprehensive|complete|concrete|direct)",
     "straight": r"(most|more) (straightforward|typical|common|standard|likely intended)",
     "expects": r"(question|they|exam|test|problem) (likely|probably|might|may)? ?(expects?|wants?|intends?|is looking for)",
     "mcq": r"(multiple[- ]choice|in such questions|typical (exam|test|mcq)|answer key|test[- ]maker)",
     "broad": r"(too (broad|general|vague))",
     "safe": r"(safe(st)? (answer|choice|bet))"}
fin = [r for r in rows if not r["cut"]]
for name, pat in H.items():
    rg = re.compile(pat, re.I); out = {0: [0, 0], 1: [0, 0]}
    late = {0: 0, 1: 0}
    for r in fin:
        t = r["think"]; ms = [m.start() for m in rg.finditer(t)]
        out[r["correct"]][0] += 1; out[r["correct"]][1] += bool(ms)
        if ms and ms[-1] > 0.7 * len(t): late[r["correct"]] += 1
    nr, nw = out[1][0], out[0][0]
    print(f"{name:9s} any: right {out[1][1]}/{nr} ({100*out[1][1]/nr:.0f}%) wrong {out[0][1]}/{nw} ({100*out[0][1]/nw:.0f}%) | in last 30%: right {late[1]} wrong {late[0]}")
