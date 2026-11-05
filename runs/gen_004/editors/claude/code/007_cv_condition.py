# Which layer's MLP activations give a commit condition s = v.a that is high at commit points and ~0 at paragraph-end controls?
# v = mean(commit) - mean(control), orthogonalised to mean(control) so controls average 0. 2-fold CV, 20 random splits.
import json, numpy as np
from sklearn.metrics import roc_auc_score
pts = json.load(open("work/read1_points.json"))
cm = np.array([p["kind"] == "commit" for p in pts])
rng = np.random.default_rng(0)
for l in range(10, 24):
    a = np.load(f"work/fwd_01/mlp{l}.npy").astype(np.float64)
    aucs, sep = [], []
    for _ in range(20):
        f = rng.random(len(pts)) < 0.5
        mc, mt = a[f & cm].mean(0), a[f & ~cm].mean(0)
        v = mc - mt; v -= (v @ mt) / (mt @ mt) * mt
        s = a[~f] @ v
        aucs.append(roc_auc_score(cm[~f], s))
        sep.append((s[cm[~f]].mean() - s[~cm[~f]].mean()) / s[~cm[~f]].std())
    print(f"L{l:2d}: CV AUC {np.mean(aucs):.3f}  separation (commit-control)/sd(control) {np.mean(sep):.2f}")
