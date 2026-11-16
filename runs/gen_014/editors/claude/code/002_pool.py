# Hypothesis: there are enough finished-wrong verbal samples (passk bank + 96-q bank) for a causal insertion test at scale; check trace formats/lengths.
import json, collections, re
from wsp.editor import *
Q = questions()
QUANT = {"business","chemistry","physics","math","engineering"}
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pkids = {r["id"] for r in pk}
extra = [dict(r, sample=0) for r in bk if r["id"] not in pkids]
# is bank sample0 == passk sample0?
b0 = {r["id"]: r for r in bk}
same = sum(1 for r in pk if r["sample"] == 0 and b0[r["id"]]["raw"] == r["raw"])
print("bank s0 identical to passk s0:", same, "/ 64; extra bank-only questions:", len(extra))
allr = pk + extra
verb = [r for r in allr if Q[r["id"]]["category"] not in QUANT]
fw = [r for r in verb if not r["cut"] and not r["correct"]]
fr = [r for r in verb if not r["cut"] and r["correct"]]
print("verbal samples", len(verb), "finished-wrong", len(fw), "over", len({r['id'] for r in fw}), "questions; finished-right", len(fr))
import numpy as np
print("tokens fw median", np.median([r["tokens"] for r in fw]), "fr median", np.median([r["tokens"] for r in fr]))
print("fw questions:", sorted({r['id'] for r in fw}))
