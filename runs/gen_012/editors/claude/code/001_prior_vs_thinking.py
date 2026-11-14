# H1 (free, reuses gen-11 claude pass 3): does thinking OVERRIDE a correct pre-thinking belief? Compare the forced prior answer (right after <think>)
# with the pass@4 sample outcomes on 64 passk questions: count questions where the prior is right but samples go wrong, and P(gold) at the prior.
import json, numpy as np
from wsp.editor import *
G = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/"
ix = json.load(open(G + "probe3_index.json")); D = G + "fwd_03/"
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy"); LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"
Q = questions()
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
by = {}
for r in rows: by.setdefault(r["id"], []).append(r)
out = []
for t, i in enumerate(ix):
    if i["kind"] != "prior": continue
    p = np.zeros(10)
    for a, b in zip(tk[t], tl[t]):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    p /= p.sum(); g = Q[i["id"]]["answer_index"]
    rs = by[i["id"]]; k = sum(r["correct"] for r in rs); fin = [r for r in rs if not r["cut"]]
    out.append((i["id"], p[g], int((p > p[g]).sum()), int(p.argmax() == g), k, len(rs), sum(r["correct"] for r in fin), len(fin)))
o = np.array([x[1:] for x in out], float)
print("questions", len(o), "prior acc", o[:, 2].mean().round(3), "mean P(gold) prior", o[:, 0].mean().round(3), "sample acc", (o[:, 3].sum() / o[:, 4].sum()).round(3))
for lab, m in [("prior right", o[:, 2] == 1), ("prior wrong", o[:, 2] == 0)]:
    print(f"{lab}: n {m.sum():.0f}  sample acc {o[m, 3].sum() / o[m, 4].sum():.3f}  finished acc {o[m, 5].sum() / max(1, o[m, 6].sum()):.3f}")
# by pass@4 group
for lab, m in [("never right", o[:, 3] == 0), ("mixed", (o[:, 3] > 0) & (o[:, 3] < o[:, 4])), ("always right", o[:, 3] == o[:, 4])]:
    print(f"{lab:12s} n {m.sum():.0f}  prior acc {o[m, 2].mean():.3f}  P(gold) {o[m, 0].mean():.3f}  gold rank {o[m, 1].mean():.2f}")
print("prior right but samples mostly wrong (k<=1):", [x[0] for x in out if x[3] == 1 and x[4] <= 1])
