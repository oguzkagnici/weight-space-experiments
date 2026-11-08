# Hypothesis: the L23 n881 commit-opener unit adds accuracy on top of the carried stack (L22 two-finalist gate + ed_a L23
# short answer): stack+unit (passes 3/4) vs carry alone (passes 6/5) on the same bank questions, paired, same seeds.
import json, sys, numpy as np
d = json.load(open("work/dev_ids.json"))
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl")}
def load(k):
    try: return {json.loads(l)["id"]: json.loads(l) for l in open(f"work/fwd_{k:02d}/gen_{k}_edited.jsonl")}
    except FileNotFoundError: return {}
U = {**load(3), **load(4)}; C = {**load(6), **load(5)}
def row(name, ids, A, Bm, la, lb):
    ids = [q for q in ids if q in A and q in Bm]
    if not ids: print(f"{name}: no data"); return
    x = np.array([A[q]["correct"] - Bm[q]["correct"] for q in ids], float); n = len(ids)
    se = x.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"{name:8s} {lb} {sum(Bm[q]['correct'] for q in ids)} -> {la} {sum(A[q]['correct'] for q in ids)} of {n} | "
          f"diff {100*x.mean():+.1f} pp (SE {100*se:.1f}, z {x.mean()/se if se>0 else 0:.2f}) | +{(x>0).sum()}/-{(x<0).sum()} | "
          f"cut {sum(Bm[q]['cut'] for q in ids)} -> {sum(A[q]['cut'] for q in ids)} | identical {sum(A[q]['raw']==Bm[q]['raw'] for q in ids)}")
sets = {"DEV32": d["dev"], "EXTRA16": d["extra16"], "REST48": d["rest48"], "ALL": d["dev"] + d["extra16"] + d["rest48"]}
for lbl, A, Bm, la, lb in (("unit vs carry", U, C, "stack+unit", "carry"), ("carry vs bank", C, B, "carry", "parent")):
    print("==", lbl)
    for s, ids in sets.items(): row(s, ids, A, Bm, la, lb)
if "-v" in sys.argv:
    for q in sets["ALL"]:
        if q in U and q in C and U[q]["correct"] != C[q]["correct"]:
            print(f"  {q} carry {C[q]['correct']} cut {C[q]['cut']} pred {C[q]['pred']} | unit {U[q]['correct']} cut {U[q]['cut']} pred {U[q]['pred']} | gold {U[q]['gold']}")
