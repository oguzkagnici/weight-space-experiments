# Hypothesis: E1 (L22 H12/H15 query off the first-in-list key direction) changes conclusions only through finalist picks; measure paired overall and both-finish results on holdout48 at 8,192 against the parent bank, plus first divergence and finalist-order statistics.
import json, sys, glob, re
import numpy as np
SH = "/home/user/weight-space-experiments/runs/shared/"
bank = {json.loads(l)["id"]: json.loads(l) for l in open(SH + "holdout48_bank_gen008champ_cap8192_det1.jsonl")}
rows = {}
for d in sys.argv[1:]:
    for f in glob.glob(d + "/gen_*_edited.jsonl"):
        for l in open(f):
            r = json.loads(l); rows[r["id"]] = r
ids = [i for i in rows if i in bank]
print("n", len(ids))
def paired(sel):
    d = np.array([rows[i]["correct"] - bank[i]["correct"] for i in sel])
    return 100 * d.mean(), 100 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0, int((d > 0).sum()), int((d < 0).sum()), len(d)
pa = np.mean([bank[i]["correct"] for i in ids]) * 100; ea = np.mean([rows[i]["correct"] for i in ids]) * 100
print("overall: parent %.1f -> edited %.1f ; diff %+.2f (SE %.2f) +%d/-%d n %d" % ((pa, ea) + paired(ids)))
bf = [i for i in ids if not bank[i]["cut"] and not rows[i]["cut"]]
print("both-finish: parent %.1f -> edited %.1f ; diff %+.2f (SE %.2f) +%d/-%d n %d" % ((np.mean([bank[i]["correct"] for i in bf]) * 100, np.mean([rows[i]["correct"] for i in bf]) * 100) + paired(bf)))
print("cut parent %d edited %d ; mean tokens %.0f -> %.0f" % (sum(bank[i]["cut"] for i in ids), sum(rows[i]["cut"] for i in ids), np.mean([bank[i]["tokens"] for i in ids]), np.mean([rows[i]["tokens"] for i in ids])))
fd = []
ident = 0
for i in ids:
    a, b = bank[i]["raw"], rows[i]["raw"]
    if a == b: ident += 1; continue
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    fd.append(k / max(1, len(a)))
print("identical %d ; first difference median %.1f%% of parent trace" % (ident, 100 * np.median(fd) if fd else 0))
ans_change = [i for i in bf if bank[i]["pred"] != rows[i]["pred"]]
print("both-finish answer changed:", len(ans_change), [(i[-5:], bank[i]["pred"], rows[i]["pred"], bank[i]["gold"]) for i in ans_change])
# last pair statement: pick first- vs second-named
pat = re.compile(r"\b(?:either|between)\s+(?:options?\s+)?\(?([A-J])\)?\s+(?:or|and)\s+\(?([A-J])\)?(?![A-Za-z])")
for nm, src in (("parent", bank), ("edited", rows)):
    c = {"pick1st": 0, "pick2nd": 0, "other": 0}
    for i in ids:
        r = src[i]
        if r["cut"] or not r["pred"]: continue
        think = r["raw"].split("</think>")[0]
        m = list(pat.finditer(think))
        if not m: continue
        a, b = m[-1].group(1), m[-1].group(2)
        c["pick1st" if r["pred"] == a else "pick2nd" if r["pred"] == b else "other"] += 1
    print(nm, "last either/between pair statement (finished):", c)
