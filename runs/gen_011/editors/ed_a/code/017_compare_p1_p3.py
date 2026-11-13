# Hypothesis: E1 lowers the first-named advantage (total and H12 direct) and the attention gap between first- and second-named pair letters, and does not lower P(gold|pair).
import json, sys
import numpy as np
from wsp.editor import *
DO, DE = "work/fwd_01", sys.argv[1]
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
L = "ABCDEFGHIJ"
E = load_original("model.embed_tokens.weight")[LETTER_IDS].numpy(); gF = load_original("model.embedding_norm.weight").numpy()
kind = np.array([p["kind"] for p in probes])
def logits(D):
    h = np.load(D + "/h24.npy").astype(np.float32); s = 1 / np.sqrt((h ** 2).mean(1) + 1e-5)
    return (h * gF) @ E.T * s[:, None], s
def oe(M):
    return np.array([((M[2*i, L.index(p['a'])] - M[2*i, L.index(p['b'])]) - (M[2*i+1, L.index(p['a'])] - M[2*i+1, L.index(p['b'])])) / 2 for i, p in enumerate(probes)])
def comp(D, l, nm, s):
    c = np.load(f"{D}/out_model.layers.{l}.{nm}.npy").astype(np.float32); return (c * gF) @ E.T * s[:, None]
def h12(D, s):
    x = np.load(f"{D}/in_model.layers.22.self_attn.o_proj.npy").astype(np.float32)
    Wo = load_original("model.layers.22.self_attn.o_proj.weight").numpy(); gp = load_original("model.layers.22.post_operator_norm.weight").numpy()
    y = x @ Wo.T; ry = 1 / np.sqrt((y ** 2).mean(1) + 1e-5)
    out = {}
    for h in (12, 13, 14, 15):
        out[h] = (gp * (x[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T) * ry[:, None] * gF) @ E.T * s[:, None]
    return out
def pgold(M):
    r = []
    for i, p in enumerate(probes):
        a, b = L.index(p["a"]), L.index(p["b"]); g = L.index(p["gold"])
        for j in (2*i, 2*i+1):
            if g in (a, b):
                z = M[j, [a, b]] / 0.6; z = np.exp(z - z.max()); z /= z.sum(); r.append((p["kind"], z[[a, b].index(g)], int(M[j, g] == M[j].max())))
    return r
def attn(D):
    off = np.load(D + "/offsets_tok.npy"); A = np.load(D + "/attn_last_L22.npy").astype(np.float32)
    out = []
    for t, (f, s_, T) in enumerate(rows):
        a = A[off[t]:off[t+1]]; out.append([a[f-1:f+2, 12].sum(), a[s_-1:s_+2, 12].sum(), a[f-1:f+2, 15].sum(), a[s_-1:s_+2, 15].sum()])
    return np.array(out)
R = {}
for tag, D in (("orig", DO), ("edit", DE)):
    M, s = logits(D); H = h12(D, s)
    R[tag] = dict(oe=oe(M), h12=oe(H[12]), h15=oe(H[15]), h13=oe(H[13]), h14=oe(H[14]), mix22=oe(comp(D, 22, "post_operator_norm", s)), mlp22=oe(comp(D, 22, "post_ffn_norm", s)), mlp23=oe(comp(D, 23, "post_ffn_norm", s)), M=M, pg=pgold(M), att=attn(D))
for k in ["oe", "h12", "h15", "h13", "h14", "mix22", "mlp22", "mlp23"]:
    line = []
    for kd in ["real", "syn"]:
        m = kind == kd; o, e = R["orig"][k][m], R["edit"][k][m]; d = e - o
        line.append(f"{kd}: {o.mean():+.3f} -> {e.mean():+.3f} (d {d.mean():+.3f} se {d.std()/np.sqrt(m.sum()):.3f})")
    print(f"{k:6s}", " | ".join(line))
for kd in ["real", "syn"]:
    o = [x for x in R["orig"]["pg"] if x[0] == kd]; e = [x for x in R["edit"]["pg"] if x[0] == kd]
    d = np.array([b[1] - a[1] for a, b in zip(o, e)])
    print(f"P(gold|pair, T=0.6) {kd}: {np.mean([x[1] for x in o]):.4f} -> {np.mean([x[1] for x in e]):.4f} (d {d.mean():+.4f} se {d.std()/np.sqrt(len(d)):.4f}); gold top letter {sum(x[2] for x in o)} -> {sum(x[2] for x in e)} of {len(o)}")
for kd in ["real", "syn"]:
    m = np.repeat(kind == kd, 2)
    print(f"attn windows {kd}: H12 first {R['orig']['att'][m,0].mean():.3f}->{R['edit']['att'][m,0].mean():.3f} second {R['orig']['att'][m,1].mean():.3f}->{R['edit']['att'][m,1].mean():.3f} | H15 first {R['orig']['att'][m,2].mean():.3f}->{R['edit']['att'][m,2].mean():.3f} second {R['orig']['att'][m,3].mean():.3f}->{R['edit']['att'][m,3].mean():.3f}")
# flips of the pair pick under swap
def flips(M):
    return sum(np.sign(M[2*i, L.index(p['a'])] - M[2*i, L.index(p['b'])]) != np.sign(M[2*i+1, L.index(p['a'])] - M[2*i+1, L.index(p['b'])]) for i, p in enumerate(probes) if p["kind"] == "real")
print("real pick flips under swap", flips(R["orig"]["M"]), "->", flips(R["edit"]["M"]), "of", (kind == "real").sum())
top_o = R["orig"]["M"].argmax(1); top_e = R["edit"]["M"].argmax(1)
print("top letter changed in", (top_o != top_e).sum(), "of", len(top_o), "texts")
