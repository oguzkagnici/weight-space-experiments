# Hypothesis: in finished traces with several in-think statements, the first statement already equals the final answer
# (so closing at a firm first statement costs little), while hedged first statements ("likely X or Y", "maybe") change.
import json, re
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl"
R = [json.loads(l) for l in open(BANK)]
ans = re.compile(r"answer is:?\s*\(?\*?\*?([A-J])\b\)?")
hedge = re.compile(r"(likely|maybe|perhaps|probably|might|suspect|think|lean|guess)\W*$", re.I)
same = diff = 0; rows = []
for r in R:
    if r["cut"]: continue
    ms = list(ans.finditer(r["think"]))
    if len(ms) < 2: continue
    pre = r["think"][max(0, ms[0].start() - 40):ms[0].start()]
    h = bool(re.search(r"likely|maybe|perhaps|probably|might|suspect|I think|lean|guess|or \(?[A-J]\)?\?", pre + r["think"][ms[0].end():ms[0].end() + 20], re.I))
    s = ms[0].group(1) == r["pred"]
    rows.append((s, h, ms[0].group(1) == r["gold"], r["correct"], round(ms[0].start() / len(r["think"]), 2), round((len(r['think']) - ms[0].start()) / 4)))
for h in (False, True):
    sub = [x for x in rows if x[1] == h]
    if sub:
        print(f"hedged={h}: n {len(sub)} first==final {sum(x[0] for x in sub)} first right {sum(x[2] for x in sub)} final right {sum(x[3] for x in sub)} "
              f"median pos {sorted(x[4] for x in sub)[len(sub)//2]} median ~tokens after first {sorted(x[5] for x in sub)[len(sub)//2]}")
