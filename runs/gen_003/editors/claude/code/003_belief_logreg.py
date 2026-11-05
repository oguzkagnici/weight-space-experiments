# hypothesis H6 check: is the option-end belief stronger than the mean-difference probe suggests? Logistic regression
# (L2, PCA-128 features fitted on the training fold), 2-fold CV by question; pick = argmax probe score within question.
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
d = json.load(open("work/option_ends.json")); qs = d["questions"]
off = np.load("work/fwd_01/offsets.npy")
rng = np.random.default_rng(0); fold = rng.permutation(len(qs)) % 2
for l in (10, 12, 14, 15):
    H = np.load(f"work/fwd_01/h{l}.npy", mmap_mode="r")
    X, y, g = [], [], []
    for i, q in enumerate(qs):
        e = np.array(d["ends"][q]["ends"]) + off[i]; v = H[e].astype(np.float64); v -= v.mean(0)
        X.append(v); y += [int(j == d["ends"][q]["answer"]) for j in range(len(e))]; g += [i] * len(e)
    X, y, g = np.concatenate(X), np.array(y), np.array(g)
    for C in (0.01, 0.1):
        accs = []
        for f in (0, 1):
            tr = fold[g] != f; te = ~tr
            p = PCA(128).fit(X[tr]); clf = LogisticRegression(C=C, max_iter=2000).fit(p.transform(X[tr]), y[tr])
            s = clf.decision_function(p.transform(X[te]))
            for i in np.unique(g[te]):
                m = g[te] == i; accs.append(int(np.argmax(s[m]) == np.argmax(y[te][m])))
        print(f"h{l} C={C}: CV probe acc {np.mean(accs):.3f} (n={len(accs)})")
