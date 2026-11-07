# Paired analysis of an edited-only generate pass against the parent's trace bank (byte-identical to the parent under
# det1, checked 32/32). Hypothesis tested (edit v1): gains come from cut traces that narrow to two finalists; untouched
# traces stay close to the parent; after two-finalist paragraphs the edited model opens with a commit word more often.
# Usage: ./py code/011_paired_vs_bank.py <pass k> [ids-json-key ...]
import sys, json, re, math, collections
from wsp.editor import tok
k = int(sys.argv[1])
BANK = "/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl"
bank = {json.loads(l)["id"]: json.loads(l) for l in open(BANK)}
E = [json.loads(l) for l in open(f"work/fwd_{k:02d}/gen_{k}_edited.jsonl")]
g = json.load(open("work/groups.json")); dv = json.load(open("work/dev_ids.json"))
grp = {q: n for n in ("narrow", "fin_narrow") for q in g[n]}
L = r"\(?([A-J])\)?"
pair = re.compile(r"\b(?:either|between)\s+(?:option\s+|options\s+)?" + L + r"\s*(?:or|and|vs\.?)\s+(?:option\s+)?" + L + r"\b"
                  r"|\b(?:answer|choice|options?)\s+(?:is|are|could be|might be|seems|likely)?\s*(?:likely\s+)?" + L + r"\s+or\s+" + L + r"\b"
                  r"|\b" + L + r"\s+(?:or|vs\.?)\s+" + L + r"\s*[\?\.]", re.I)
COMMIT = ("Thus", "Therefore", "So", "Hence")
def body(r): return r["raw"][len("<think>"):] if r["raw"].startswith("<think>") else r["raw"]
def mech(text):
    t = text.split("</think>")[0]; n = c = 0
    for m in pair.finditer(t):
        if len({x for x in m.groups() if x}) != 2: continue
        e = t.find("\n\n", m.end())
        if e < 0: continue
        n += 1; w = re.match(r"[A-Za-z(\-]+", t[e + 2:e + 42].lstrip("*# ")); c += bool(w and w.group(0) in COMMIT)
    return n, c
rows = []; tot = collections.Counter()
for r in E:
    b = bank[r["id"]]; tb, te = body(b), body(r)
    same = tb == te
    if same: fd = 1.0
    else:
        i = next((j for j in range(min(len(tb), len(te))) if tb[j] != te[j]), min(len(tb), len(te)))
        fd = len(tok(tb[:i], add_special_tokens=False)["input_ids"]) / max(1, b["tokens"])
    nb, cb = mech(tb); ne, ce = mech(te)
    for key, v in (("pairs_parent", nb), ("commit_parent", cb), ("pairs_edit", ne), ("commit_edit", ce)): tot[key] += v
    set_ = "dev" if r["id"] in dv["dev"] else "extra16"
    rows.append(dict(id=r["id"], set=set_, grp=grp.get(r["id"], "other"), pc=b["correct"], ec=r["correct"], pcut=b["cut"], ecut=r["cut"],
                     same=same, fd=round(fd, 3), pred=(b["pred"], r["pred"]), gold=r["gold"]))
for name, sel in (("DEV32", [x for x in rows if x["set"] == "dev"]), ("EXTRA16", [x for x in rows if x["set"] == "extra16"]), ("ALL48", rows)):
    n = len(sel); d = [x["ec"] - x["pc"] for x in sel]; m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) if n > 1 else 0
    fds = sorted(x["fd"] for x in sel if not x["same"])
    print(f"{name}: parent {sum(x['pc'] for x in sel)} -> edited {sum(x['ec'] for x in sel)} of {n} | diff {100*m:+.1f} pp (SE {100*se:.1f}, z {m/se if se else 0:.2f}) "
          f"| +{sum(v > 0 for v in d)}/-{sum(v < 0 for v in d)} | cut {sum(x['pcut'] for x in sel)} -> {sum(x['ecut'] for x in sel)} "
          f"| byte-identical {sum(x['same'] for x in sel)}/{n} | first-divergence median {fds[len(fds)//2] if fds else '-'}")
print("mechanism:", dict(tot))
for x in rows:
    if x["pc"] != x["ec"] or x["pcut"] != x["ecut"]:
        print(f"  {x['id']:14s} {x['set']:7s} {x['grp']:10s} correct {x['pc']}->{x['ec']} cut {x['pcut']}->{x['ecut']} pred {x['pred']} gold {x['gold']} fd {x['fd']}")
json.dump(rows, open(f"work/paired_p{k}.json", "w"), indent=0)
