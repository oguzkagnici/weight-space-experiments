# Paired analysis of edited generate passes vs the parent bank: overall and both-finish paired diff (SE), W->R / R->W, cut, tokens, first diff,
# and predicted-letter counts. Usage: py code/017_paired.py dev 04 | hold 05 06 | hb 07 08
import json, re, sys, numpy as np
from wsp.editor import *
G = re.compile(r"$^")
mode, ks = sys.argv[1], sys.argv[2:]
bank = {"dev": "passk_gen008champ_cap8192_det1.jsonl", "hold": "holdout48_bank_gen008champ_cap8192_det1.jsonl", "hb": "holdout_bank_gen008champ_cap3072_det1.jsonl"}[mode]
P = {(r["id"], r.get("sample", 0)): r for r in map(json.loads, open("/home/user/weight-space-experiments/runs/shared/" + bank))}
E = [json.loads(l) for k in ks for l in open(f"work/fwd_{k}/gen_{int(k)}_edited.jsonl")]
rows = []
for e in E:
    p = P[(e["id"], e.get("sample", 0))]
    fd = next((i for i, (a, b) in enumerate(zip(p["raw"], e["raw"])) if a != b), min(len(p["raw"]), len(e["raw"])))
    rows.append(dict(id=e["id"], s=e.get("sample", 0), pc=p["correct"], ec=e["correct"], pcut=p["cut"], ecut=e["cut"], pt=p["tokens"], et=e["tokens"],
                     same=p["raw"] == e["raw"], fd=fd, fdf=fd / max(1, len(p["raw"])), pg=len(G.findall(p["think"])), eg=len(G.findall(e["think"])), pp=p["pred"], ep=e["pred"]))
def rep(R, name):
    d = np.array([r["ec"] - r["pc"] for r in R], float); n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    print(f"{name}: n {n}  parent {100*np.mean([r['pc'] for r in R]):.1f}%  edited {100*np.mean([r['ec'] for r in R]):.1f}%  diff {100*d.mean():+.2f} pts (SE {100*se:.2f})  W->R {int((d>0).sum())}  R->W {int((d<0).sum())}")
rep(rows, "OVERALL"); rep([r for r in rows if not r["pcut"] and not r["ecut"]], "BOTH-FINISH")
print(f"cut: parent {sum(r['pcut'] for r in rows)} edited {sum(r['ecut'] for r in rows)}; tokens mean parent {np.mean([r['pt'] for r in rows]):.0f} edited {np.mean([r['et'] for r in rows]):.0f}")
print(f"identical traces {sum(r['same'] for r in rows)}/{len(rows)}")
for r in rows:
    if not r["same"]:
        print(f"  {r['id']} s{r['s']}: {r['pp']}->{r['ep']} correct {r['pc']}->{r['ec']} cut {r['pcut']}->{r['ecut']} tok {r['pt']}->{r['et']} first diff at {r['fdf']:.2f}")
json.dump(rows, open(f"work/paired_{mode}.json", "w"))
from collections import Counter
print("pred letter counts parent", sorted(Counter(r["pp"] for r in rows).items(), key=lambda x: str(x[0])), "\n               edited", sorted(Counter(r["ep"] for r in rows).items(), key=lambda x: str(x[0])))
