# Build the detector->re-check unit on L23 n881 (freed by folding into its near-duplicate n331, cos .9996).
# Gate (w1 row) = K*(w - T*p)/g_ffn: w = corrupt-minus-clean mean direction of normalised h22 (input proxy of the L23 MLP), fit on TRAIN
# questions of the pass-1 pairs only; p = minimum-variance bias proxy (mean x.p = 1 over clean sentence ends); T = PCT-th percentile of the
# train corrupt scores. Up (w3 row) = ALPHA*p/g_ffn. Write (w2 col) = re-check openers ("Wait", " Wait", "Actually", " Actually") minus the
# embedding mean, through the final and post-ffn norms, scaled to fit 0.98 x the Q8 block absmax.
import json, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
from wsp.editor import *
K = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0; ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 50
P = json.load(open("work/pairs1.json")); qs = sorted({p["id"] for p in P})
rng = np.random.default_rng(7); tr_q = set(rng.choice(qs, len(qs) // 2, replace=False)); json.dump(sorted(tr_q), open("work/train_qs.json", "w"))
tr = np.array([p["id"] in tr_q for p in P])
H = np.load("work/fwd_01/h22.npy").astype(np.float64); Hn = H / np.sqrt((H ** 2).mean(1, keepdims=True))
C, X = Hn[0::2], Hn[1::2]
w = (X[tr] - C[tr]).mean(0); w /= np.linalg.norm(w)
G = C; mu = G.mean(0); Cv = np.cov(G.T); Cv += 1e-2 * np.eye(2048) * np.trace(Cv) / 2048
p = np.linalg.solve(Cv, mu); p /= (p @ mu)
T = np.percentile(X[tr] @ w / (X[tr] @ p), PCT)
pre = lambda Z: K * (Z @ w - T * (Z @ p))
for nm, m in (("train", tr), ("TEST", ~tr)):
    print(f"{nm}: fires(>0) corrupt {np.mean(pre(X[m]) > 0):.2f} clean {np.mean(pre(C[m]) > 0):.2f}; paired corrupt>clean {np.mean(pre(X[m]) > pre(C[m])):.2f}; AUC {roc_auc_score(np.r_[np.zeros(m.sum()), np.ones(m.sum())], np.r_[pre(C[m]), pre(X[m])]):.2f}")
print("pre-activation clean mean %.2f sd %.2f; corrupt mean %.2f" % (pre(C).mean(), pre(C).std(), pre(X).mean()))
g = load_original("model.layers.23.ffn_norm.weight").double().numpy()
w1 = load_original("model.layers.23.feed_forward.w1.weight"); w3 = load_original("model.layers.23.feed_forward.w3.weight"); w2 = load_original("model.layers.23.feed_forward.w2.weight")
r = float(w1[881] @ w1[331] / (w1[331] @ w1[331])) * float(w3[881] @ w3[331] / (w3[331] @ w3[331]))
w2[:, 331] += r * w2[:, 881]
w1[881] = torch.tensor(K * (w - T * p) / g, dtype=torch.float32); w3[881] = torch.tensor(ALPHA * p / g, dtype=torch.float32)
E = load("model.embed_tokens.weight").double()
ids = [tok(s, add_special_tokens=False).input_ids[0] for s in ("Wait", " Wait", "Actually", " Actually")]
tgt = E[ids].mean(0)
gf = load("model.embedding_norm.weight").double(); gp = load("model.layers.23.post_ffn_norm.weight").double()
v = (tgt - E.mean(0)) * gf / gp
b0 = (881 // 32) * 32; blk = w2[:, b0:b0 + 32].clone(); blk[:, 881 - b0] = 0; amax = blk.abs().max(1).values.double()
scale = float(torch.quantile(0.98 * amax / v.abs().clamp_min(1e-9), 0.10)); col = v * scale
col = torch.maximum(torch.minimum(col, 0.98 * amax), -0.98 * amax); w2[:, 881] = col.float()
print("fold ratio %.4f | w2 col norm %.3f | targets %s" % (r, float(col.norm()), ids))
for n, tn in (("model.layers.23.feed_forward.w1.weight", w1), ("model.layers.23.feed_forward.w3.weight", w3), ("model.layers.23.feed_forward.w2.weight", w2)): save_edit(n, tn)
json.dump({"K": K, "ALPHA": ALPHA, "PCT": PCT, "T": float(T)}, open("work/edit_params.json", "w"))
