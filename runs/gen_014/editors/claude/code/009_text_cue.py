# Hypothesis (step 1a, text level): right samples mention the decisive cue's content words during option judging (after the first 20% of the trace) more than wrong samples of the same question.
import json, re, sys, collections, numpy as np
sys.path.insert(0, "code")
from wsp.editor import *
from importlib import import_module
CUES = import_module("005_cues").CUES
Q = questions()
pk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bk = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
pkids = {r["id"] for r in pk}
allr = pk + [dict(r, sample=0) for r in bk if r["id"] not in pkids]
STOP = set("the a an of to in for on and or is are was were be by with that this which who what when his her he she they it its at as from than not no all any can may following does there would will have has had been how about into after before during".split())
def words(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 2]
rows = []
for r in allr:
    q = r["id"]
    if q not in CUES or r["cut"]: continue
    opts = set(w for o in Q[q]["options"] for w in words(o))
    cw = [w for w in set(words(CUES[q])) if w not in opts] or list(set(words(CUES[q])))
    think = r["raw"][len("<think>"):].split("</think>")[0].lower()
    body = think[int(0.2 * len(think)):]
    hits = sum(1 for w in cw if re.search(r"\b%s" % re.escape(w[:max(4, len(w) - 2)]), body))
    per1k = sum(len(re.findall(r"\b%s" % re.escape(w[:max(4, len(w) - 2)]), body)) for w in cw) / max(1, len(body)) * 1000
    rows.append(dict(id=q, s=r["sample"], correct=r["correct"], frac=hits / len(cw), per1k=per1k, n=len(cw)))
import pandas as pd
df = pd.DataFrame(rows)
print(df.groupby("correct")[["frac", "per1k"]].agg(["mean", "count"]).round(3))
# within question (questions with both right and wrong finished samples)
d = []
for q, g in df.groupby("id"):
    if g.correct.nunique() == 2:
        d.append((q, g[g.correct == 1].frac.mean() - g[g.correct == 0].frac.mean(), g[g.correct == 1].per1k.mean() - g[g.correct == 0].per1k.mean()))
d = np.array([(x[1], x[2]) for x in d])
print("mixed cue questions:", len(d), "within-q diff (right - wrong) frac %.3f (SE %.3f); per1k %.3f (SE %.3f)" % (d[:, 0].mean(), d[:, 0].std(ddof=1) / np.sqrt(len(d)), d[:, 1].mean(), d[:, 1].std(ddof=1) / np.sqrt(len(d))))
print(df[df.id.isin(["mmlupro_6110", "mmlupro_2654", "mmlupro_6562", "mmlupro_6800", "mmlupro_1171"])].round(2).to_string())
