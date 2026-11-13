# H4 (look): why do finished-wrong samples of mixed questions reject the gold option? Print the context of the LAST gold mention and of the first mention.
import json, re, collections
from wsp.editor import questions
Q = questions()
P = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
mixed = json.load(open("work/mixed_ids.json"))
for r in P:
    if r["id"] not in mixed or r["correct"] or r["cut"]: continue
    q = Q[r["id"]]; g = q["answer"]; t = r["think"]
    pat = re.compile(r"\(%s\)|[Oo]ption %s\b" % (g, g))
    ms = [m.start() for m in pat.finditer(t)]
    print(f"=== {r['id']} s{r['sample']} {q['category']} gold {g} pred {r['pred']} gold-mentions {len(ms)} chars {len(t)} first@{ms[0]/len(t):.2f}" if ms else f"=== {r['id']} s{r['sample']} gold {g} pred {r['pred']} NO gold mention")
    if ms:
        print("  FIRST:", t[ms[0]:ms[0]+260].replace("\n"," | ").encode("ascii","replace").decode())
        print("  LAST :", t[ms[-1]:ms[-1]+260].replace("\n"," | ").encode("ascii","replace").decode())
