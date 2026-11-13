# H6: wrong traces often never evaluate the gold option (beyond restating the option list); coverage of options in the sweep predicts correctness.
import json, re, numpy as np, collections
from wsp.editor import questions
Q = questions()
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pk = {(r["id"], r["sample"]) for r in rows}
rows += [r for r in bank if (r["id"], 0) not in pk]
L = "ABCDEFGHIJ"
def mentions(t, X, otext):
    pat = re.compile(r"\(%s\)|[Oo]ption %s\b|\b%s\)|\b%s:" % (X, X, X, X))
    ms = [m.start() for m in pat.finditer(t)]
    # drop listing: mentions followed directly by the option text within 3 chars (restating list)
    ev = [s for s in ms if otext[:20].lower() not in t[s:s + 30].lower()] if otext else ms
    return ms, ev
res = []
for r in rows:
    if r["cut"]: continue
    q = Q[r["id"]]; t = r["think"]; n = len(q["options"]); g = q["answer_index"]
    cov = 0; gev = 0
    for i in range(n):
        ms, ev = mentions(t, L[i], q["options"][i])
        if len(ms) >= 2 or ev: cov += 1
        if i == g: gev = len(ms)
    res.append((r["correct"], cov / n, gev, q["category"], len(t)))
a = np.array([(c, cv, ge) for c, cv, ge, _, _ in res], float)
print("finished", len(res))
for k in (1, 0):
    s = a[a[:, 0] == k]
    print("right" if k else "wrong", len(s), "mean coverage", round(s[:, 1].mean(), 2), "gold mentions mean", round(s[:, 2].mean(), 1),
          "gold mentioned <=1 times", int((s[:, 2] <= 1).sum()))
# accuracy by gold mentions
for lo, hi in [(0, 1), (2, 3), (4, 6), (7, 99)]:
    s = a[(a[:, 2] >= lo) & (a[:, 2] <= hi)]
    print(f"gold mentions {lo}-{hi}: n {len(s)} acc {100*s[:,0].mean():.0f}")
