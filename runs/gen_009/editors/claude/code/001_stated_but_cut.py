# Hypothesis: cut traces that state "the answer is (X)" inside the thinking keep going because the statement is hedged
# or followed by a re-verification opener; the parent's L20 wrap-up unit (gen 4) does not close them. Bank-only analysis.
import json, re
from collections import Counter
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"
R = [json.loads(l) for l in open(BANK)]
ans = re.compile(r"answer is:?\s*\(?\*?\*?([A-J])\b\)?")
rows = []
for r in R:
    t = r["think"]; ms = list(ans.finditer(t))
    if not ms: continue
    m = ms[0]; frac = m.start() / max(1, len(t))
    pre = t[max(0, m.start() - 70):m.start()].replace("\n", "/"); post = t[m.end():m.end() + 90].replace("\n", "/")
    rows.append((r["cut"], r["correct"], r["id"], r["category"], len(ms), round(frac, 2), ms[0].group(1) == r["gold"], pre, post))
cut = [x for x in rows if x[0]]; fin = [x for x in rows if not x[0]]
print(f"traces with in-think statement: cut {len(cut)} of {sum(r['cut'] for r in R)}, finished {len(fin)} of {sum(1-r['cut'] for r in R)}")
print("finished: first statement position (median frac)", sorted(x[5] for x in fin)[len(fin)//2], "statements per trace", Counter(x[4] for x in fin).most_common(4))
for x in sorted(cut, key=lambda x: x[5]):
    print(f"\n{x[2]} {x[3][:8]} n_stmt {x[4]} first@{x[5]} first_right {x[6]}\n   ...{x[7]}[[ANS]]{x[8]}")
