# H1c: some MLP neurons detect "an answer was just stated" (active at commit points, quiet at ordinary paragraph ends).
# They are the condition for a targeted edit: adding a wrap-up direction to their output only acts after a commitment.
import json, numpy as np
from sklearn.metrics import roc_auc_score
pts = json.load(open("work/read1_points.json"))
cm = np.array([p["kind"] == "commit" for p in pts]); lab = np.array([p["label"] for p in pts])
rows = []
for l in range(24):
    a = np.load(f"work/fwd_01/mlp{l}.npy").astype(np.float32)
    mc, mt = a[cm].mean(0), a[~cm].mean(0)
    sd = a.std(0) + 1e-4
    for j in np.argsort(-(mc - mt) / sd)[:15]:
        auc = roc_auc_score(cm, a[:, j])
        # within commit: does it fire equally on reopen and close outcomes?
        r = a[cm & (lab == "reopen"), j].mean(); c = a[cm & (lab == "close"), j].mean()
        rows.append((auc, l, int(j), float(mc[j]), float(mt[j]), float(r), float(c), float(np.abs(a[~cm, j]).max())))
rows.sort(reverse=True)
print("AUC commit vs control | layer neuron | mean commit | mean control | commit&reopen | commit&close | max |control|")
for r in rows[:30]:
    print(f"  {r[0]:.3f} | L{r[1]:2d} {r[2]:5d} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.3f} | {r[6]:+.3f} | {r[7]:.3f}")
json.dump(rows, open("work/commit_detectors.json", "w"))
# residual-stream linear separability by layer (mean-difference direction, leave-one-out-free rough AUC)
print("\nResidual mean-difference direction AUC by layer (in-sample):")
for l in [4, 8, 12, 14, 16, 18, 20, 22, 24]:
    h = np.load(f"work/fwd_01/h{l}.npy").astype(np.float32)
    dvec = h[cm].mean(0) - h[~cm].mean(0)
    s = h @ dvec
    print(f"  h{l}: {roc_auc_score(cm, s):.3f}", end="")
print()
