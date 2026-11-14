# Build the test-gaming suppression unit on L23 n881 (freed by folding into its near-duplicate n331, as in gen 10).
# Gate (w1 row) = K*(w_lda - T*p): w_lda = shrinkage LDA (gaming-onset positions vs content " exam"/" test" + other tokens, pass-4 texts),
# projected off the prompt-sink inputs; p = minimum-variance bias proxy (mean x.p = 1); T = PCT-th percentile of gaming-onset scores.
# Up (w3 row) = ALPHA*p. Write (w2 col) = -g_final * (count-weighted mean E[next tokens at gaming onsets] - mean E[all-token centroid proxy]) / g_post,
# scaled so 90% of entries fit 0.98 x Q8 block absmax, rest clipped.
import json, sys, numpy as np, torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from wsp.editor import *
K = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; PCT = float(sys.argv[3]) if len(sys.argv) > 3 else 40
items = json.load(open("work/read4_index.json")); meta = json.load(open("work/fwd_04/meta.json"))
X = np.load("work/fwd_04/in_model.layers.23.feed_forward.w1.npy", mmap_mode="r"); offs = np.cumsum([0] + meta["n_tokens"])
rng = np.random.default_rng(0); rows, lab, gen_rows, sink = [], [], [], []
for k, (it, o) in enumerate(zip(items, offs[:-1])):
    for p in it["pos"]: rows.append(o + p); lab.append(2)
    for p in it["neg_content"]: rows.append(o + p); lab.append(1)
    g = np.arange(o + it["plen"], o + it["n"]); g = g[~np.isin(g, [o + p for p in it["pos"] + it["neg_content"]])]
    s = rng.choice(g, size=min(500, len(g)), replace=False); rows += list(s); lab += [0] * len(s); gen_rows += list(s)
    sink.append([o + i for i in range(4)])
A = np.asarray(X[np.array(rows)], np.float32); lab = np.array(lab)
m = LDA(solver="lsqr", shrinkage="auto").fit(A, (lab == 2).astype(int)); w = m.coef_[0].astype(np.float64)
G = np.asarray(X[np.array(gen_rows)], np.float64); mu = G.mean(0)
C = np.cov(G.T); C += 1e-3 * np.eye(C.shape[0]) * np.trace(C) / C.shape[0]
p = np.linalg.solve(C, mu); p /= (p @ mu)
S = np.asarray(X[np.array(sink).ravel()], np.float64); Qs, _ = np.linalg.qr(S.T); w = w - Qs @ (Qs.T @ w)
sh = A[lab == 2] @ w; T = np.percentile(sh, PCT)
pre = lambda Z: K * (Z @ w - T * (Z @ p))
print("bias proxy sd", round(float((G @ p).std()), 3), "| fires(>0): gaming", round(float((pre(A[lab == 2]) > 0).mean()), 2),
      "content", round(float((pre(A[lab == 1]) > 0).mean()), 3), "other", round(float((pre(A[lab == 0]) > 0).mean()), 4), "sink max", round(float(pre(S).max()), 2))
w1 = load_original("model.layers.23.feed_forward.w1.weight"); w3 = load_original("model.layers.23.feed_forward.w3.weight"); w2 = load_original("model.layers.23.feed_forward.w2.weight")
r = float(w1[881] @ w1[331] / (w1[331] @ w1[331])) * float(w3[881] @ w3[331] / (w3[331] @ w3[331]))
w2[:, 331] += r * w2[:, 881]
w1[881] = torch.tensor(K * (w - T * p), dtype=torch.float32); w3[881] = torch.tensor(ALPHA * p, dtype=torch.float32)
E = load("model.embed_tokens.weight").double()
nxt = {int(k): v for k, v in json.load(open("work/gaming_next_ext.json")).items()}
ids = list(nxt); wts = torch.tensor([nxt[i] for i in ids], dtype=torch.float64); wts /= wts.sum()
tgt = (E[ids] * wts[:, None]).sum(0)
gf = load("model.embedding_norm.weight").double(); gp = load("model.layers.23.post_ffn_norm.weight").double()
v = -(tgt - E.mean(0)) * gf / gp
b0 = (881 // 32) * 32; blk = w2[:, b0:b0 + 32].clone(); blk[:, 881 - b0] = 0; amax = blk.abs().max(1).values.double()
scale = float(torch.quantile(0.98 * amax / v.abs().clamp_min(1e-9), 0.10)); col = v * scale
col = torch.maximum(torch.minimum(col, 0.98 * amax), -0.98 * amax); w2[:, 881] = col.float()
print("fold ratio", round(r, 4), "| w2 col norm", round(float(col.norm()), 3), "| targets", [(tok.decode([i]), round(float(x), 2)) for i, x in zip(ids, wts)][:8])
for n, tn in (("model.layers.23.feed_forward.w1.weight", w1), ("model.layers.23.feed_forward.w3.weight", w3), ("model.layers.23.feed_forward.w2.weight", w2)): save_edit(n, tn)
json.dump({"K": K, "ALPHA": ALPHA, "PCT": PCT, "T": float(T)}, open("work/edit_params.json", "w"))
