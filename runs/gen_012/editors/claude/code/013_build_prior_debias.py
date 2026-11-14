# H9 edit: correct the content-free letter preference of the forced-answer readout INSIDE reasoning (pre-thinking prior and mid-trace forced
# probes) via L23 n7528, which is active at those readouts (mean .50 at prior probes, .48 at mid-trace probes) but silent at final-answer copy
# positions (.00). Offsets c_L = log(mean P_L) - log(gold share) from 320 prior probes (pass 2); add to w2[:, 7528] the minimal-norm
# pre-norm MLP-output vector giving logit shift -S*c (centered over letters) at the mean activation (frozen norms from gen-11 pass-3 probes).
import sys, json, numpy as np, torch
from wsp.editor import *
S = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; JS = [7528, 6877, 4830, 1228, 4441, 2650]
D2 = "work/fwd_02/"; ix = json.load(open("work/probe2_index.json")); LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"
tk = np.load(D2 + "topk_ids.npy"); tl = np.load(D2 + "topk_logprobs.npy")
P, G = [], []
for t, i in enumerate(ix):
    if i["kind"] != "prior": continue
    p = np.zeros(10)
    for a, b in zip(tk[t], tl[t]):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    P.append(p / p.sum()); G.append(LET.index(i["gold"]))
P = np.array(P); G = np.array(G)
c = np.log(P.mean(0)) - np.log((np.bincount(G, minlength=10) + 1) / (len(G) + 10)); c -= c.mean()
print("c =", np.round(c, 2).tolist())
G3 = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/"
ix3 = json.load(open(G3 + "probe3_index.json")); pm = np.array([i["kind"] == "prior" for i in ix3])
h23 = np.load(G3 + "fwd_03/mlp23.npy")[pm]; H24 = np.load(G3 + "fwd_03/h24.npy")[pm]
W2 = load_original("model.layers.23.feed_forward.w2.weight").numpy(); gp = load("model.layers.23.post_ffn_norm.weight").numpy()
gf = load("model.embedding_norm.weight").numpy(); E = load("model.embed_tokens.weight")[LI].numpy()
rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)
ry = rms(h23 @ W2.T).mean(); r24 = rms(H24).mean()
M = E * (gp * gf / (ry * r24))[None, :]; Mc = M - M.mean(0, keepdims=True)
v = np.linalg.pinv(Mc) @ (-S * c)                          # total pre-norm MLP-output change wanted at the readout
W = load_original("model.layers.23.feed_forward.w2.weight").clone(); tot = np.zeros(2048); clipped = 0
for J in JS:
    act = h23[:, J].mean(); u = v * act / sum(h23[:, k].mean() ** 2 for k in JS); col = W2[:, J]
    b0 = (J // 32) * 32; amax = np.abs(W2[:, b0:b0 + 32]).max(1)     # stay within each row's Q8_0 block absmax (no re-quantisation)
    new = col + u; clipped += int((np.abs(col + u) > 0.98 * amax).sum())
    tot += (new - col) * act; W[:, J] = torch.tensor(new, dtype=W.dtype)
    print(f"n{J}: act {act:.3f} |col| {np.linalg.norm(col):.3f} |u| {np.linalg.norm(u):.3f}")
pred = (tot * gp / ry * gf / r24) @ E.T; pred -= pred.mean()
print("clipped entries", clipped, "| predicted centered letter shift at mean act:", np.round(pred, 2).tolist(), "| wanted", np.round(-S * c, 2).tolist())
save_edit("model.layers.23.feed_forward.w2.weight", W)
json.dump({"S": S, "JS": JS, "c": c.tolist(), "clipped": clipped, "pred": pred.tolist()}, open("work/edit_params.json", "w"))
