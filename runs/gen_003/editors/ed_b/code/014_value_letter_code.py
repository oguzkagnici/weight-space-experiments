# Hypothesis: when H12 attends to an option's end/newline token, the value it copies encodes that option's letter; for long options the letter code may be blurred, so the final answer deviates from H12's attention choice. Check the letter readout of H12's per-option value (v -> o_proj head 12 -> tied letter rows, through final-norm gain) vs option length.
import numpy as np, pickle, torch
from wsp.editor import load, LETTER_IDS
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb"))
off = np.load(D + "offsets_tok.npy")
V = np.load(D + "out_model.layers.22.self_attn.v_proj.npy", mmap_mode="r")
A = np.load(D + "attn_last_L22.npy")
rows = pickle.load(open("work/diag03.pkl", "rb"))
Wo = load("model.layers.22.self_attn.o_proj.weight")[:, 12 * 128:13 * 128].numpy().astype(np.float64)
E = load("model.embed_tokens.weight")[LETTER_IDS].numpy().astype(np.float64)
gf = load("model.embedding_norm.weight").numpy().astype(np.float64)
gp = load("model.layers.22.post_operator_norm.weight").numpy().astype(np.float64)
R = (E * gf * gp) @ Wo  # 10 x 128: letter readout of a head-12 value (up to positive scale)
R = R - R.mean(0)
dec, lens, marg = [], [], []
dis = []
for i, o in enumerate(sp):
    a, b = off[i], off[i + 1]
    v = np.asarray(V[a:b, 3 * 128:4 * 128]).astype(np.float64); att = A[a:b, 12].astype(np.float64)
    n = o["n"]
    for j, (s0, s1) in enumerate(o["spans"]):
        t = list(range(s0, s1 + 2)); w = att[t] / att[t].sum()
        vv = w @ v[t]
        sc = R[:n] @ vv
        dec.append(np.argmax(sc) == j); lens.append(s1 - s0 + 2); marg.append(sc[j] - np.max(np.delete(sc, j)))
    # H12 output letter readout vs its attention pick
    out = att @ v; sc = R[:n] @ out
    r = rows[i]; dis.append((np.argmax(sc), r["massarg"], r["pred"], r["gold"]))
dec = np.array(dec); lens = np.array(lens); marg = np.array(marg)
for lo, hi in [(0, 6), (6, 12), (12, 25), (25, 1000)]:
    m = (lens >= lo) & (lens < hi)
    print("option len %3d-%3d: n %4d letter decoded %.3f  mean margin %.2f" % (lo, hi, m.sum(), dec[m].mean(), marg[m].mean()))
dis = np.array(dis)
print("H12 output letter == H12 attention pick %.3f; == final pred %.3f; == gold %.3f" % ((dis[:, 0] == dis[:, 1]).mean(), (dis[:, 0] == dis[:, 2]).mean(), (dis[:, 0] == dis[:, 3]).mean()))
