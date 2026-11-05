# Hypothesis: short causal convs (kernel 7) and local attention mix the previous option's features into an option's end/newline token, so the belief leaks to the neighbour: errors would over-pick gold+1 (or gold-1). Compare the distribution of pred-gold among errors with a position-matched null (pred letter distribution independent of gold).
import numpy as np, json, pickle
rows = [json.loads(l) for l in open("work/fwd_02/score_2_original.jsonl")]
d3 = pickle.load(open("work/diag03.pkl", "rb")); sp = pickle.load(open("work/spans240.pkl", "rb"))
R = [(r["answer"], r["pred"], len(r["probs"]), r["probs"]) for r in rows]
R += [(o["gold"], x["pred"], o["n"], None) for o, x in zip(sp, d3)]
R = [r for r in R if r[2] == 10]
g = np.array([r[0] for r in R]); p = np.array([r[1] for r in R])
err = g != p
obs = np.bincount((p - g)[err] + 9, minlength=19)
# null: shuffle preds among errors keeping gold, excluding equal (permutation null)
rng = np.random.default_rng(0); null = np.zeros(19)
pe = p[err]; ge = g[err]
for k in range(2000):
    pp = rng.permutation(pe); m = pp != ge
    null += np.bincount((pp - ge)[m] + 9, minlength=19) / m.sum() * err.sum()
null /= 2000
for d in range(-4, 5):
    print("pred-gold %+d: observed %3d  null %.1f" % (d, obs[d + 9], null[d + 9]))
# second choice analysis using probs (DEV2 only): when wrong, rank of gold+1/gold-1
