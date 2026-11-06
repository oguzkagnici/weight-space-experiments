# Compare passes 4 (carry, target -9) and 5 (target -15) on DEV32, paired against the parent and against each other.
import json, numpy as np
G = json.load(open("work/groups.json"))
grp = {q: g for g, v in G.items() for q in v}
def L(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
O = L("work/fwd_04/gen_4_original.jsonl"); A = L("work/fwd_04/gen_4_edited.jsonl"); B = L("work/fwd_05/gen_5_edited.jsonl")
ids = json.load(open("work/dev32.json"))
def fd(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: return n
        n += 1
    return n
for nm, E in (("carry -9", A), ("dose -15", B)):
    d = np.array([E[q]["correct"] - O[q]["correct"] for q in ids])
    print(f"{nm}: parent {sum(O[q]['correct'] for q in ids)} edited {sum(E[q]['correct'] for q in ids)}  diff {100*d.mean():+.1f} pp (SE {100*d.std(ddof=1)/np.sqrt(len(d)):.1f}) +{(d>0).sum()}/-{(d<0).sum()}"
          f"  cut {sum(O[q]['cut'] for q in ids)}->{sum(E[q]['cut'] for q in ids)}  tokens {np.mean([E[q]['tokens']-O[q]['tokens'] for q in ids]):+.0f}"
          f"  identical {sum(E[q]['raw']==O[q]['raw'] for q in ids)}")
    for g in ("cut_stmt", "cut_nostmt", "finished"):
        q2 = [q for q in ids if grp[q] == g]
        print(f"   {g:10s} n={len(q2):2d} correct {sum(O[q]['correct'] for q in q2)}->{sum(E[q]['correct'] for q in q2)} cut {sum(O[q]['cut'] for q in q2)}->{sum(E[q]['cut'] for q in q2)}"
              f" median first-diff char {int(np.median([fd(O[q]['raw'], E[q]['raw']) for q in q2]))}")
d = np.array([B[q]["correct"] - A[q]["correct"] for q in ids])
print(f"dose -15 vs carry: {100*d.mean():+.1f} pp +{(d>0).sum()}/-{(d<0).sum()}")
for q in ids:
    r = (O[q]["correct"], A[q]["correct"], B[q]["correct"])
    if len(set(r)) > 1: print("  ", q, grp[q], "parent/carry/d15 correct", r, "cut", (O[q]["cut"], A[q]["cut"], B[q]["cut"]))
# is the parent side identical to the bank (cache consistency)?
bank = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
print("parent == bank raw:", sum(O[q]["raw"] == bank[q]["raw"] for q in ids), "/ 32")
