# Hypothesis check: is any category anomalously weak for a format reason (e.g. health 21% vs biology 63%)? Combine DEV[:240] (pass 1 lens) and DEV2 (pass 2) per category with gold/pred letter positions and option counts.
import numpy as np, json, pickle
from wsp.editor import questions
Q = questions()
rows = [json.loads(l) for l in open("work/fwd_02/score_2_original.jsonl")]
d3 = pickle.load(open("work/diag03.pkl", "rb")); sp = pickle.load(open("work/spans240.pkl", "rb"))
recs = [(r["category"], r["answer"], r["pred"], len(r["probs"]), r["id"]) for r in rows] + [(o["cat"], o["gold"], x["pred"], o["n"], o["id"]) for o, x in zip(sp, d3)]
cats = sorted(set(r[0] for r in recs))
for c in cats:
    R = [r for r in recs if r[0] == c]
    gl = np.array([r[1] for r in R]); pr = np.array([r[2] for r in R]); no = np.array([r[3] for r in R])
    print("%-17s n %3d acc %.2f  mean gold pos %.1f  mean pred pos %.1f  frac10opt %.2f" % (c, len(R), (gl == pr).mean(), gl.mean(), pr.mean(), (no == 10).mean()))
