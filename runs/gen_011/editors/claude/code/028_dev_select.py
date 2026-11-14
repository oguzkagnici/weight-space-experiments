# DEV set for pass 6: passk questions whose parent samples use the gaming phrase most (G regex), parent cached at cap 8,192 x 4.
import json, re, collections
G = re.compile(r"(typical (exam|test|mcq|multiple)|in such (questions|exams)|test[- ]?maker|exam (style|question)s? (often|usually|typically|would|likely|expect)|(exam|question|test) (likely|probably) expects?)", re.I)
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
by = collections.defaultdict(list)
for r in rows: by[r["id"]].append(r)
sc = []
for q, rs in by.items():
    ng = sum(bool(G.search(r["think"])) for r in rs); acc = sum(r["correct"] for r in rs); cut = sum(r["cut"] for r in rs)
    sc.append((ng, -acc, q, acc, cut))
sc.sort(reverse=True)
for s in sc[:12]: print(s)
dev = [s[2] for s in sc[:6]]; print("DEV", dev)
json.dump({"type": "generate", "model": "edited", "questions": dev, "n": 4, "cap": 8192}, open("work/req_dev.json", "w"))
