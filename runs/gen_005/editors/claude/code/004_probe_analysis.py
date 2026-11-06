# Analyse pass 1: forced-answer letter distribution at 30% / 65% of cut traces.
import json, numpy as np
from wsp.editor import *
LETTERS = "ABCDEFGHIJ"
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
pts = json.load(open("work/probe_points.json"))
ids = np.load("work/fwd_01/topk_ids.npy"); lp = np.load("work/fwd_01/topk_logprobs.npy")
# letter token ids after "(": check tokenization of "(A"
LID = [tok.encode("is (" + l, add_special_tokens=False)[-1] for l in LETTERS]
print("letter ids after '(':", LID[:4], "LETTER_IDS:", LETTER_IDS[:4])

rows = []
for k, p in enumerate(pts):
    b = B[p["id"]]; gold = b["gold"]
    nopt = len(b.get("options", [])) or 10
    pr = np.zeros(10)
    for t, l in zip(ids[k], lp[k]):
        if t in LID: pr[LID.index(t)] += np.exp(l)
    mass = pr.sum(); pn = pr / max(mass, 1e-9)
    am = int(pn.argmax())
    rows.append(dict(p, mass=mass, conf=pn[am], arg=LETTERS[am], gold=gold, ok=LETTERS[am] == gold, pgold=pn[LETTERS.index(gold)], final_ok=b.get("correct")))
import collections
for g in ("nostmt", "stmt"):
    for f in (0.30, 0.65):
        R = [r for r in rows if r["group"] == g and r["frac"] == f]
        if not R: continue
        acc = np.mean([r["ok"] for r in R]); conf = np.mean([r["conf"] for r in R]); pg = np.mean([r["pgold"] for r in R])
        hi = [r for r in R if r["conf"] >= 0.8]
        print(f"{g:6s} {f:.2f} n={len(R):2d} letter-mass={np.mean([r['mass'] for r in R]):.2f} argmax-acc={acc:.2f} mean-conf={conf:.2f} p(gold)={pg:.2f} | conf>=.8: n={len(hi)} acc={np.mean([r['ok'] for r in hi]) if hi else float('nan'):.2f}")
# calibration over all
print("calibration (all points):")
for lo, hi_ in ((0, .4), (.4, .6), (.6, .8), (.8, .95), (.95, 1.01)):
    R = [r for r in rows if lo <= r["conf"] < hi_]
    print(f"  conf [{lo:.2f},{hi_:.2f}) n={len(R):2d} acc={np.mean([r['ok'] for r in R]) if R else float('nan'):.2f}")
# stability across 30->65 for same question
byq = collections.defaultdict(dict)
for r in rows: byq[r["id"]][r["frac"]] = r
same = [(d[0.30]["arg"] == d[0.65]["arg"], d[0.65]["ok"]) for d in byq.values() if len(d) == 2]
print("same argmax at 30% and 65%:", sum(s for s, _ in same), "/", len(same), "; acc when same:", np.mean([o for s, o in same if s]), "when different:", np.mean([o for s, o in same if not s]))
json.dump(rows, open("work/probe_rows.json", "w"), default=float, indent=0)
