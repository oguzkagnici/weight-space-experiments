# Hypothesis: at the end of the first option sweep, finished traces name a winner, while cut traces end the sweep with
# several options still "possible" (no single winner) and restart. Print the text between the last option of sweep 1
# and the next sweep's start (ASCII, <= 400 chars) and count the options marked plausible after sweep 1.
import json, re
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
rows = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
opt_pat = re.compile(r"(?:^|\n)\s*(?:\*\*)?(?:Option\s+)?\(?([A-J])\)?(?:\*\*)?\s*[:\.\)-]")
A = json.load(open("work/lh_structure.json"))
for r in A:
    t = rows[r["id"]]["think"].replace("<think>", "")
    heads = [(m.start(), m.group(1)) for m in opt_pat.finditer(t)]
    if not heads: continue
    end1 = None
    for i in range(1, len(heads)):
        if heads[i][1] <= "B" and heads[i - 1][1] >= "D":
            end1 = i; break
    if end1 is None:
        seg = t[heads[-1][0]:][:900]
    else:
        # from the start of the last option paragraph of sweep 1 to the start of sweep 2
        seg = t[heads[end1 - 1][0]:heads[end1][0]]
        seg = seg[-900:]
    para_end = seg.split("\n\n")
    tail = "\n\n".join(para_end[1:])[:500] if len(para_end) > 1 else seg[-400:]
    print(f"=== {r['id']} {r['cat']} cut={r['cut']} ok={r['correct']} sweep1_end_frac={(heads[end1][0]/len(t)) if end1 else 1:.2f}")
    print(tail.encode("ascii", "replace").decode().replace("\n", " | ")[:500])
