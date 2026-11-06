# HOLDOUT32 result for the -15 dose (pass 6), paired against the parent.
import json, numpy as np
def L(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
O = L("work/fwd_06/gen_6_original.jsonl"); E = L("work/fwd_06/gen_6_edited.jsonl"); ids = json.load(open("work/holdout32.json"))
d = np.array([E[q]["correct"] - O[q]["correct"] for q in ids])
print(f"HOLDOUT32 dose -15: parent {sum(O[q]['correct'] for q in ids)} edited {sum(E[q]['correct'] for q in ids)} diff {100*d.mean():+.1f} pp (SE {100*d.std(ddof=1)/np.sqrt(32):.1f}) +{(d>0).sum()}/-{(d<0).sum()}"
      f" cut {sum(O[q]['cut'] for q in ids)}->{sum(E[q]['cut'] for q in ids)} tokens {np.mean([E[q]['tokens']-O[q]['tokens'] for q in ids]):+.0f} identical {sum(E[q]['raw']==O[q]['raw'] for q in ids)}")
for q in ids:
    if O[q]["correct"] != E[q]["correct"]: print("  ", q, O[q]["category"], "parent", O[q]["correct"], O[q]["cut"], "-> edited", E[q]["correct"], E[q]["cut"])
