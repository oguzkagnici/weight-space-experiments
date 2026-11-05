# Hypothesis: (diagnostic) exact direct-logit attribution of the letter stage on DEV2 (512 q): stream h22, each L22 head, MLP22, L23 mixer, MLP23, to see which components write the early-letter bias (pred C 98 vs gold 54, J 17 vs 37) and which flip answers.
import numpy as np, json, torch
from wsp.editor import load, LETTER_IDS
D = "work/fwd_02/score_2_acts/original/"
ld = lambda f: np.load(D + f + ".npy").astype(np.float64)
rows = [json.loads(l) for l in open("work/fwd_02/score_2_original.jsonl")]
n = len(rows); g = np.array([r["answer"] for r in rows]); pred = np.array([r["pred"] for r in rows]); nopt = np.array([len(r["probs"]) for r in rows])
E = load("model.embed_tokens.weight")[LETTER_IDS].numpy().astype(np.float64)
gf = load("model.embedding_norm.weight").numpy().astype(np.float64)
h = {l: ld("h%d" % l) for l in [20, 21, 22, 23, 24]}
rf = np.sqrt((h[24] ** 2).mean(1) + 1e-5)  # final norm scale per question
RE = E * gf  # 10 x 2048
def lens(x):  # x: n x 2048 residual-space contribution -> letter logits (n x 10), using the final rms of each question
    return (x @ RE.T) / rf[:, None]
L = lens(h[24])
ok = np.array([np.argmax(L[i, :nopt[i]]) == pred[i] for i in range(n)])
print("lens(h24) reproduces pred: %.3f; acc %.4f" % (ok.mean(), (pred == g).mean()))
# components
Wo = load("model.layers.22.self_attn.o_proj.weight").numpy().astype(np.float64)
oin = ld("in_model.layers.22.self_attn.o_proj"); poin = ld("in_model.layers.22.post_operator_norm")
print("o_proj(in) vs post_operator_norm input: rel err %.4f" % (np.linalg.norm(oin @ Wo.T - poin) / np.linalg.norm(poin)))
gpo = load("model.layers.22.post_operator_norm.weight").numpy().astype(np.float64)
rpo = np.sqrt((poin ** 2).mean(1) + 1e-5)
heads = np.stack([lens(gpo * (oin[:, k * 128:(k + 1) * 128] @ Wo[:, k * 128:(k + 1) * 128].T) / rpo[:, None]) for k in range(16)])  # 16 x n x 10
attn22 = h[22] * 0 + (gpo * poin / rpo[:, None])
mlp22 = h[23] - h[22] - attn22
comps = {"h22": lens(h[22]), "attn22": lens(attn22), "mlp22": lens(mlp22), "L23": lens(h[24] - h[23])}
def center(x):
    out = np.zeros_like(x)
    for i in range(n):
        k = nopt[i]; out[i, :k] = x[i, :k] - x[i, :k].mean()
    return out
ten = nopt == 10
print("centered mean letter logits on 10-option q (A..J):")
for k, v in list(comps.items()) + [("final", L)]:
    c = center(v); print("  %-7s" % k, np.round(c[ten].mean(0), 2), " gold-help %.2f" % np.mean([c[i, g[i]] for i in range(n)]))
for k in range(16):
    c = center(heads[k]); gh = np.mean([c[i, g[i]] for i in range(n)])
    if abs(gh) > 0.1 or np.abs(c[ten].mean(0)).max() > 0.3:
        print("  head%-3d" % k, np.round(c[ten].mean(0), 2), " gold-help %.2f" % gh)
# stage accuracy
for k, v in [("h22", comps["h22"]), ("+attn22", comps["h22"] + comps["attn22"]), ("+mlp22", comps["h22"] + comps["attn22"] + comps["mlp22"]), ("final", L)]:
    print("acc %-8s %.3f" % (k, np.mean([np.argmax(v[i, :nopt[i]]) == g[i] for i in range(n)])))
np.save("work/dla02_heads.npy", heads); np.savez("work/dla02_comps.npz", **comps, final=L, rf=rf)
