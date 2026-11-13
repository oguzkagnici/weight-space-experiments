# H21: the test-gaming heuristic proper ("typical exam/test (style|question|answer)", "in such questions", "test maker", "exam question ... expects") marks/causes wrong picks.
import json, re, numpy as np, collections
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pk = {(r["id"], r["sample"]) for r in rows}; rows += [r for r in bank if (r["id"], 0) not in pk]
G = re.compile(r"(typical (exam|test|mcq|multiple)|in such (questions|exams)|test[- ]?maker|exam (style|question)s? (often|usually|typically|would|likely|expect)|(exam|question|test) (likely|probably) expects?)", re.I)
tab = collections.Counter(); byq = collections.defaultdict(list); pos = {0: [], 1: []}
for r in rows:
    if r["cut"]: continue
    t = r["think"]; ms = [m.start() / len(t) for m in G.finditer(t)]
    g = bool(ms); tab[(g, r["correct"])] += 1; byq[r["id"]].append((g, r["correct"]))
    if ms: pos[r["correct"]].append(ms[0])
for g in (True, False):
    a, b = tab[(g, 1)], tab[(g, 0)]; print(f"gaming={g}: n {a+b} acc {100*a/max(1,a+b):.1f}")
d = [np.mean([c for gg, c in v if gg]) - np.mean([c for gg, c in v if not gg]) for v in byq.values() if any(gg for gg, _ in v) and any(not gg for gg, _ in v)]
print("within-question diff", round(np.mean(d), 3), "SE", round(np.std(d, ddof=1) / np.sqrt(len(d)), 3), "n q", len(d))
print("first position median right", np.median(pos[1]).round(2) if pos[1] else None, "wrong", np.median(pos[0]).round(2) if pos[0] else None)
cut = [r for r in rows if r["cut"]]; print("cut traces with gaming:", sum(bool(G.search(r["think"])) for r in cut), "/", len(cut))
