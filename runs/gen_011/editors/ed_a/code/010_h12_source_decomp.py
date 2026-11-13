# Hypothesis: H12's first-named advantage comes mostly from attending more to the first-named letter in the pair statement (source = pair span), not from prompt option lines; decompose H12's direct letter-logit order effect by source position.
import json, sys
import numpy as np
from wsp.editor import *
D1, D2 = "work/fwd_01", "work/fwd_02"
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
off = np.load(D1 + "/offsets_tok.npy"); att = np.load(D1 + "/attn_last_L22.npy")[:, 12].astype(np.float32)
V = np.load(D2 + "/out_model.layers.22.self_attn.v_proj.npy", mmap_mode="r")
off2 = np.load(D2 + "/offsets.npy"); assert (off2 == off).all()
x = np.load(D1 + "/in_model.layers.22.self_attn.o_proj.npy").astype(np.float32)
Wo = load("model.layers.22.self_attn.o_proj.weight").numpy(); gp = load("model.layers.22.post_operator_norm.weight").numpy()
E = load("model.embed_tokens.weight")[LETTER_IDS].numpy(); gF = load("model.embedding_norm.weight").numpy()
h24 = np.load(D1 + "/h24.npy").astype(np.float32); s24 = 1 / np.sqrt((h24 ** 2).mean(1) + 1e-5)
y = x @ Wo.T; ry = 1 / np.sqrt((y ** 2).mean(1) + 1e-5)
U = (E * gF * gp) @ Wo[:, 12 * 128:13 * 128]     # 10 x 128
L = "ABCDEFGHIJ"
cats = ["pair_first", "pair_second", "pair_other", "prompt", "think_rest", "suffix"]
C = np.zeros((len(rows), len(cats), 10)); M = np.zeros((len(rows), len(cats)))
gates = []
for t, (f, s_, T) in enumerate(rows):
    p = probes[t // 2]
    a = att[off[t]:off[t + 1]]; v = V[off[t]:off[t + 1], 3 * 128:4 * 128].astype(np.float32)
    av = a @ v; g = (x[t, 12 * 128:13 * 128] @ av) / (av @ av); gates.append(g)
    contrib = (a[:, None] * (v @ U.T)) * g * ry[t] * s24[t]       # T x 10
    pl = len(tok(render_think(p["id"]), add_special_tokens=False)["input_ids"])
    lab = np.full(T, 4)
    lab[:pl] = 3
    lab[T - 9:] = 5
    lo, hi = min(f, s_) - 3, max(f, s_) + 3
    lab[lo:hi] = 2
    lab[f - 1:f + 2] = 0; lab[s_ - 1:s_ + 2] = 1
    for c in range(6):
        C[t, c] = contrib[lab == c].sum(0); M[t, c] = a[lab == c].sum()
print("gate mean %.3f" % np.mean(gates))
def oe(Mx):
    out = []
    for i, p in enumerate(probes):
        a, b = L.index(p["a"]), L.index(p["b"])
        out.append(((Mx[2*i, a] - Mx[2*i, b]) - (Mx[2*i+1, a] - Mx[2*i+1, b])) / 2)
    return np.array(out)
kind = np.array([p["kind"] for p in probes])
tot = oe(C.sum(1))
print("H12 total order effect: real %.3f syn %.3f" % (tot[kind == "real"].mean(), tot[kind == "syn"].mean()))
for c, nm in enumerate(cats):
    e = oe(C[:, c])
    # first/second categories swap roles between orig and swap texts -> report attention mass
    print(f"{nm:12s} real {e[kind=='real'].mean():+.3f} syn {e[kind=='syn'].mean():+.3f} | attn mass real {M[kind.repeat(2)=='real', c].mean():.3f} syn {M[kind.repeat(2)=='syn', c].mean():.3f}")
# within prompt: split by option line of the first- vs second-named letter
np.save("work/p2_C.npy", C)
