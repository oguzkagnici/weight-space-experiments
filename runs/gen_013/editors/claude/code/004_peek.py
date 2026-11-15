# Read-only helper: show a practice question and the openings of chosen pass@4 samples (diagnose the first wrong step by hand).
import json, sys
from wsp.editor import questions
Q = questions(); qid = sys.argv[1]; ss = [int(x) for x in sys.argv[2].split(",")]; a, b = int(sys.argv[3]), int(sys.argv[4])
pk = {(r["id"], r["sample"]): r for r in map(json.loads, open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl"))}
q = Q[qid]; print(q["question"][:700].encode("ascii", "replace").decode()); print([o[:80] for o in q["options"]], "gold", q["answer"])
for s in ss:
    r = pk[(qid, s)]; t = r["think"]; print(f"--- s{s} pred {r['pred']} correct {r['correct']} len {len(t)}"); print(t[a:b].encode("ascii", "replace").decode())
