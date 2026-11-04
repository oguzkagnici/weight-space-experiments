# Hypothesis: the q-v1 edit improves more than a constant letter offset would (content routing, not only bias). Compare with cross-fitted per-letter debias of each model.
import sys, json, numpy as np
k = int(sys.argv[1])
def rd(m):
    return {r["id"]: r for r in map(json.loads, open(f"work/fwd_{k:02d}/score_{k}_{m}.jsonl"))}
o, e = rd("original"), rd("edited"); ids = [i for i in o if i in e]
np.set_printoptions(precision=2, suppress=True, linewidth=200)
def arr(d):
    P = np.full((len(ids), 10), -1e9)
    for r, i in enumerate(ids):
        pr = np.array(d[i]["probs"]); P[r, :len(pr)] = np.log(np.maximum(pr, 1e-9))
    g = np.array([d[i]["answer"] for i in ids])
    return P, g
Lo, g = arr(o); Le, _ = arr(e)
nopt = np.array([len(o[i]["probs"]) for i in ids])
ten = nopt == 10
for nm, L in [("orig", Lo), ("edit", Le)]:
    c = L[ten] - L[ten].mean(1, keepdims=True)
    print(nm, "mean centred letter logp (10-opt)", c.mean(0))
def debias_acc(L, rng):
    idx = np.where(ten)[0]; acc = []
    for rep in range(20):
        p = rng.permutation(idx); a, b = p[:len(p)//2], p[len(p)//2:]
        tot = 0
        for tr, te in [(a, b), (b, a)]:
            c = L[tr] - L[tr].mean(1, keepdims=True); off = c.mean(0)
            tot += np.sum((L[te] - off).argmax(1) == g[te])
        acc.append(tot / len(idx))
    return np.mean(acc)
rng = np.random.default_rng(0)
print("10-opt n", ten.sum(), " acc orig %.3f edit %.3f" % (np.mean(Lo[ten].argmax(1) == g[ten]), np.mean(Le[ten].argmax(1) == g[ten])))
print("cross-fitted debiased acc: orig %.3f edit %.3f" % (debias_acc(Lo, rng), debias_acc(Le, rng)))
# per gold-letter accuracy
for nm, L in [("orig", Lo), ("edit", Le)]:
    print(nm, "acc by gold letter", np.array([np.mean(L[g == j].argmax(1) == j) for j in range(10)]))
