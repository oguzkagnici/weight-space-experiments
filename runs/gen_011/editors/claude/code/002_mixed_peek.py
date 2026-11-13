# H2 (look): how do right and wrong samples of the same mixed question judge options in the sweep?
import json, collections, re, sys
from wsp.editor import questions
Q = questions()
P = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
by = collections.defaultdict(list)
for r in P: by[r["id"]].append(r)
mixed = [q for q, rs in by.items() if 0 < sum(r["correct"] for r in rs) < 4]
print(len(mixed), mixed)
json.dump(mixed, open("work/mixed_ids.json", "w"))
qid = sys.argv[1] if len(sys.argv) > 1 else mixed[0]
q = Q[qid]; print(qid, q["category"], "gold", q["answer"]); print(q["question"][:300].encode("ascii","replace").decode())
for i,o in enumerate(q["options"]): print(" ", "ABCDEFGHIJ"[i], o[:90].encode("ascii","replace").decode())
for r in sorted(by[qid], key=lambda r: r["sample"]):
    t = r["think"]
    print(f"--- s{r['sample']} pred {r['pred']} correct {r['correct']} cut {r['cut']} tokens {r['tokens']} chars {len(t)}")
    print(t[:1400].encode("ascii","replace").decode().replace("\n"," | "))
