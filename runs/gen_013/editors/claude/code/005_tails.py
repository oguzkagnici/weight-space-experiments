# Read-only: question stem + the last ~450 chars of thinking of each sample (what decided the pick), for hand classification of wrong samples.
import json, sys
from wsp.editor import questions
Q = questions()
pk = {(r["id"], r["sample"]): r for r in map(json.loads, open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl"))}
for qid in sys.argv[1:]:
    q = Q[qid]; print("=" * 20, qid, q["question"][:250].encode("ascii", "replace").decode(), "| gold", q["answer"], q["options"][q["answer_index"]][:70])
    for s in range(4):
        r = pk[(qid, s)]
        if r["cut"]: continue
        print(f"  s{s} {r['pred']}{'+' if r['correct'] else '-'}: ...", r["think"][-450:].replace("\n", " ").encode("ascii", "replace").decode())
