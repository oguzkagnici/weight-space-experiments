# Analysis of pass 1 (hypothesis in code/005): forced-answer letter distribution at two-finalist paragraph ends.
# Letter logits = final-norm(h24[last]) @ E[LETTER_IDS]^T (tied output layer; one matrix product).
import json, glob, os
import numpy as np, torch
from wsp.editor import load, LETTER_IDS
D = "work/fwd_01"
meta = json.load(open("work/probe1_meta.json"))
files = sorted(glob.glob(f"{D}/*.npy")); print([os.path.basename(f) for f in files][:10])
h = np.load([f for f in files if "24" in os.path.basename(f)][0]).astype(np.float32)
assert h.shape[0] == len(meta), (h.shape, len(meta))
w = load("model.embedding_norm.weight").numpy()
E = load("model.embed_tokens.weight")[LETTER_IDS].numpy()
print("norm weight mean", float(w.mean()))
hn = h / np.sqrt((h ** 2).mean(1, keepdims=True) + 1e-5) * w
logits = hn @ E.T
p = torch.softmax(torch.tensor(logits), 1).numpy()
L = "ABCDEFGHIJ"
rows = []
for m, pr in zip(meta, p):
    am = L[int(pr.argmax())]
    r = {**m, "argmax": am, "p_gold": float(pr[L.index(m["gold"])]), "conf": float(pr.max())}
    if m.get("pair"):
        a, b = m["pair"]; pa, pb = pr[L.index(a)], pr[L.index(b)]
        r.update({"p_pair": float(pa + pb), "pick": a if pa >= pb else b, "in_pair": am in m["pair"]})
    rows.append(r)
json.dump(rows, open("work/probe1_rows.json", "w"))
def show(xs, name):
    if not xs: return
    n = len(xs)
    print(f"{name}: n={n} argmax correct {sum(x['argmax']==x['gold'] for x in xs)} | mean p_gold {np.mean([x['p_gold'] for x in xs]):.2f} "
          f"| mean conf {np.mean([x['conf'] for x in xs]):.2f}" +
          (f" | argmax in pair {sum(x['in_pair'] for x in xs)} | mean p_pair {np.mean([x['p_pair'] for x in xs]):.2f}" if xs[0].get("pair") else ""))
probes = [r for r in rows if not r.get("control")]
first = {}
for r in probes:
    if r["id"] not in first: first[r["id"]] = r
F = list(first.values())
show([r for r in F if r["cut"]], "CUT, first pair mention")
show([r for r in F if r["cut"] and r["gold"] in r["pair"]], "  ... gold in pair")
show([r for r in F if r["cut"] and r["gold"] not in r["pair"]], "  ... gold not in pair")
show([r for r in probes if r["cut"]], "CUT, all pair mentions")
show([r for r in F if not r["cut"]], "FINISHED, first pair mention")
show([r for r in rows if r.get("control")], "CUT, control at end of trace")
print("\nper trace (first mention -> end control):")
ctrl = {r["id"]: r for r in rows if r.get("control")}
for r in F:
    c = ctrl.get(r["id"])
    print(f" {r['id']:14s} cut={r['cut']} gold={r['gold']} pair={''.join(r['pair'])} @{r['frac']:.2f} argmax={r['argmax']} conf={r['conf']:.2f} "
          f"p_gold={r['p_gold']:.2f} p_pair={r['p_pair']:.2f}" + (f" | end: argmax={c['argmax']} p_gold={c['p_gold']:.2f}" if c else ""))
