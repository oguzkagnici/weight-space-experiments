# E2: L23 n881 detector->re-check unit with the gate refit on real background positions (code/015: w_in, p in w1-input space, projected
# off the sink subspace; T = Q-quantile of background scores). n881 folded into n331 first. Up = ALPHA*p. Write = re-check openers (as E1).
import json, sys, numpy as np, torch
from wsp.editor import *
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
w_in, p = np.load("work/gate_vectors.npy"); prm = json.load(open("work/gate_refit.json")); K, T = prm["K"], prm["T"]
w1 = load_original("model.layers.23.feed_forward.w1.weight"); w3 = load_original("model.layers.23.feed_forward.w3.weight"); w2 = load_original("model.layers.23.feed_forward.w2.weight")
r = float(w1[881] @ w1[331] / (w1[331] @ w1[331])) * float(w3[881] @ w3[331] / (w3[331] @ w3[331]))
w2[:, 331] += r * w2[:, 881]
w1[881] = torch.tensor(K * (w_in - T * p), dtype=torch.float32); w3[881] = torch.tensor(ALPHA * p, dtype=torch.float32)
E = load("model.embed_tokens.weight").double()
ids = [tok(s, add_special_tokens=False).input_ids[0] for s in ("Wait", " Wait", "Actually", " Actually")]
gf = load("model.embedding_norm.weight").double(); gp = load("model.layers.23.post_ffn_norm.weight").double()
v = (E[ids].mean(0) - E.mean(0)) * gf / gp
b0 = (881 // 32) * 32; blk = w2[:, b0:b0 + 32].clone(); blk[:, 881 - b0] = 0; amax = blk.abs().max(1).values.double()
scale = float(torch.quantile(0.98 * amax / v.abs().clamp_min(1e-9), 0.10)); col = torch.maximum(torch.minimum(v * scale, 0.98 * amax), -0.98 * amax)
w2[:, 881] = col.float()
print("Q", prm["Q"], "K", K, "T %.3f" % T, "ALPHA", ALPHA, "fold %.4f" % r, "col norm %.3f" % float(col.norm()))
for n, tn in (("model.layers.23.feed_forward.w1.weight", w1), ("model.layers.23.feed_forward.w3.weight", w3), ("model.layers.23.feed_forward.w2.weight", w2)): save_edit(n, tn)
json.dump({**prm, "ALPHA": ALPHA, "edit": "E2"}, open("work/edit_params.json", "w"))
