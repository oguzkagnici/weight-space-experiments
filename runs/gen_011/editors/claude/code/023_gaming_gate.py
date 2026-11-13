# H23: a shrinkage-LDA direction at the L23 MLP input separates gaming-onset positions (next token starts "exam/test/MCQ/questions/maker" in a
# gaming phrase) from content uses of " exam"/" test" and from other tokens; grouped 8-fold CV estimate of firing rates; next-token stats at positives.
import json, numpy as np, collections
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score
from wsp.editor import *
items = json.load(open("work/read4_index.json")); meta = json.load(open("work/fwd_04/meta.json"))
X = np.load("work/fwd_04/in_model.layers.23.feed_forward.w1.npy", mmap_mode="r"); offs = np.cumsum([0] + meta["n_tokens"])
req = json.load(open("work/req_read4.json"))
for it, t, n in zip(items, req["texts"], meta["n_tokens"]):
    assert len(tok(t, add_special_tokens=False).input_ids) == n, (it["id"], n)
rng = np.random.default_rng(0); rows, lab, grp = [], [], []; nxt = collections.Counter()
for k, (it, o) in enumerate(zip(items, offs[:-1])):
    ids = tok(req["texts"][k], add_special_tokens=False).input_ids
    for p in it["pos"]: rows.append(o + p); lab.append(2); grp.append(k); nxt[ids[p + 1]] += 1
    for p in it["neg_content"]: rows.append(o + p); lab.append(1); grp.append(k)
    g = np.arange(o + it["plen"], o + it["n"]); g = g[~np.isin(g, [o + p for p in it["pos"] + it["neg_content"]])]
    for p in rng.choice(g, size=min(500, len(g)), replace=False): rows.append(p); lab.append(0); grp.append(k)
print("next tokens at positives:", [(tok.decode([i]), c) for i, c in nxt.most_common(10)])
A = np.asarray(X[np.array(rows)], np.float32); lab = np.array(lab); grp = np.array(grp)
sc = np.zeros(len(A)); folds = np.array_split(np.random.default_rng(1).permutation(len(items)), 8)
for f in folds:
    te = np.isin(grp, f); m = LDA(solver="lsqr", shrinkage="auto").fit(A[~te], (lab[~te] == 2).astype(int)); sc[te] = m.decision_function(A[te])
print("held-out AUC gaming vs content-exam/test", round(roc_auc_score(lab[lab > 0] == 2, sc[lab > 0]), 3), "gaming vs other", round(roc_auc_score(lab[lab != 1] == 2, sc[lab != 1]), 4))
for pct in (30, 50, 70):
    th = np.percentile(sc[lab == 2], pct)
    print(f"threshold keeping {100-pct}% of gaming: content fires {np.mean(sc[lab==1]>th):.3f}  other fires {np.mean(sc[lab==0]>th):.4f}")
json.dump({str(k): v for k, v in nxt.items()}, open("work/gaming_next.json", "w"))
