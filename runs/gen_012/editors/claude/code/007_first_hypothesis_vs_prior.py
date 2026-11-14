# H6 (free): does the FIRST letter hypothesis stated in a trace come from the pre-thinking prior readout (anchoring path)? Compare the first
# "answer is (likely) (X)"-type statement in passk traces with the prior argmax (gen-11 pass-3 prior probes), and outcome by agreement.
import json, re, numpy as np
from wsp.editor import *
G = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/"
ix = json.load(open(G + "probe3_index.json")); tk = np.load(G + "fwd_03/topk_ids.npy"); tl = np.load(G + "fwd_03/topk_logprobs.npy")
LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"; Q = questions()
prior = {}
for t, i in enumerate(ix):
    if i["kind"] != "prior": continue
    p = np.zeros(10)
    for a, b in zip(tk[t], tl[t]):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    prior[i["id"]] = p / p.sum()
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
HYP = re.compile(r"(?:answer|option|choice) (?:is|would be|might be|could be|seems to be)(?: likely| probably)?:? \(?([A-J])\)")
agree = []; pos = []
for r in rows:
    th = r["think"] or ""; m = HYP.search(th)
    if not m or r["id"] not in prior: continue
    L = LET.index(m.group(1)); p = prior[r["id"]]
    agree.append((int(L == p.argmax()), p[L], int(m.group(1) == r["gold"]), r["correct"], int(m.group(1) == r["pred"]), m.start() / max(1, len(th))))
a = np.array(agree, float)
print("traces with a first hypothesis", len(a), "| first == prior argmax", a[:, 0].mean().round(3), "| mean prior P of first letter", a[:, 1].mean().round(3),
      "| first == gold", a[:, 2].mean().round(3), "| first == final pick", a[:, 4].mean().round(3), "| median position", np.median(a[:, 5]).round(2))
for lab, m in [("first==prior", a[:, 0] == 1), ("first!=prior", a[:, 0] == 0)]:
    print(f"  {lab}: n {m.sum():.0f}  first==gold {a[m, 2].mean():.3f}  final correct {a[m, 3].mean():.3f}  final==first {a[m, 4].mean():.3f}")
early = a[:, 5] < 0.3
print("  early first hypotheses (<30% of trace):", early.sum(), "== prior", a[early, 0].mean().round(3), "final correct", a[early, 3].mean().round(3))
