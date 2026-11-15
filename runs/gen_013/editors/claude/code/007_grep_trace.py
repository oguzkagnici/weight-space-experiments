# Read-only: find a substring in a pass@4 sample's thinking and print its offset and ~260 chars of context.
import json, sys, re
pk = {(r["id"], r["sample"]): r for r in map(json.loads, open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl"))}
qid, s, pat = sys.argv[1], int(sys.argv[2]), sys.argv[3]
t = pk[(qid, s)]["think"]
for m in list(re.finditer(pat, t))[:4]:
    print(m.start(), len(t), "|", t[max(0, m.start() - 120): m.start() + 140].replace("\n", " / ").encode("ascii", "replace").decode())
