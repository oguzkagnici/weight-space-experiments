# Paired analysis of edited-only generate passes against a parent trace bank (byte-identical to the parent under det1).
# Hypothesis (Unit A): gains come from cut traces with a firm conclusion that now close right after it; finished traces
# close sooner with the same answer; where the thinking closes, the text just before </think> is a conclusion.
# Usage: ./py code/012_paired_vs_bank.py [--holdout] <pass k> [<pass k> ...]
import sys, json, re, math
from wsp.editor import tok
args = [a for a in sys.argv[1:] if not a.startswith("--")]; HOLD = "--holdout" in sys.argv
BANK = ("/home/user/weight-space-experiments/runs/shared/holdout_bank_gen008champ_cap3072_det1.jsonl" if HOLD else
        "/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap3072_det1.jsonl")
bank = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
E = [json.loads(l) for k in args for l in open(f"work/fwd_{int(k):02d}/gen_{int(k)}_edited.jsonl")]
sp = {} if HOLD else json.load(open("work/bank_split.json")); T = set(sp.get("targets", []))
def body(r): return r["raw"][len("<think>"):] if r["raw"].startswith("<think>") else r["raw"]
rows = []
for r in E:
    b = bank[r["id"]]; tb, te = body(b), body(r); same = tb == te
    if same: fd = 1.0
    else:
        i = next((j for j in range(min(len(tb), len(te))) if tb[j] != te[j]), min(len(tb), len(te)))
        fd = len(tok(tb[:i], add_special_tokens=False)["input_ids"]) / max(1, b["tokens"])
    pre = te.split("</think>")[0][-70:] if "</think>" in te else None
    grp = "target" if r["id"] in T else ("B1" if r["id"] in sp.get("B1", []) else ("B2" if r["id"] in sp.get("B2", []) else "hold"))
    rows.append(dict(id=r["id"], grp=grp, pc=b["correct"], ec=r["correct"], pcut=b["cut"], ecut=r["cut"], same=same, fd=round(fd, 3),
                     pred=(b["pred"], r["pred"]), gold=r["gold"], tok=(b["tokens"], r["tokens"]), pre=pre))
def show(name, sel):
    n = len(sel)
    if n < 2: return
    d = [x["ec"] - x["pc"] for x in sel]; m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n)
    fds = sorted(x["fd"] for x in sel if not x["same"])
    dt = [x["tok"][1] - x["tok"][0] for x in sel]
    print(f"{name:8s}: parent {sum(x['pc'] for x in sel)} -> edited {sum(x['ec'] for x in sel)} of {n} | diff {100*m:+.1f} pp (SE {100*se:.1f}, z {m/se if se else 0:.2f}) "
          f"| +{sum(v > 0 for v in d)}/-{sum(v < 0 for v in d)} | cut {sum(x['pcut'] for x in sel)} -> {sum(x['ecut'] for x in sel)} "
          f"| tokens {sum(dt)/n:+.0f} | identical {sum(x['same'] for x in sel)}/{n} | first-div median {fds[len(fds)//2] if fds else '-'}")
for g in ("target", "B1", "B2", "hold"):
    show(g, [x for x in rows if x["grp"] == g])
show("ALL", rows); show("non-tgt", [x for x in rows if x["grp"] != "target"])
for x in rows:
    if x["pc"] != x["ec"] or x["pcut"] != x["ecut"] or x["grp"] == "target":
        print(f"  {x['id']:14s} {x['grp']:6s} correct {x['pc']}->{x['ec']} cut {x['pcut']}->{x['ecut']} pred {x['pred']} gold {x['gold']} "
              f"fd {x['fd']} tok {x['tok']} | before </think>: {(x['pre'] or '')[-55:]!r}")
json.dump(rows, open(f"work/paired_p{'_'.join(args)}.json", "w"), indent=0)
