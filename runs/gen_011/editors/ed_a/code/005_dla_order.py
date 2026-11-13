# Hypothesis: the first-named advantage at the forced commit is carried directly by a few late heads that attend more to the first-named letter token; DLA of the order effect per component and per head localises them.
import json, sys, re
import numpy as np, torch
from wsp.editor import *
D = sys.argv[1] if len(sys.argv) > 1 else "work/fwd_1"
TAG = sys.argv[2] if len(sys.argv) > 2 else "p1"
probes = json.load(open("work/p1_probes.json"))
meta = json.load(open(D + "/meta.json"))
L = "ABCDEFGHIJ"
E = load("model.embed_tokens.weight")[LETTER_IDS].numpy()  # 10 x 2048
gF = load("model.embedding_norm.weight").numpy()
h24 = np.load(D + "/h24.npy").astype(np.float32)
s = 1 / np.sqrt((h24 ** 2).mean(1) + 1e-5)
lg = (h24 * gF) @ E.T * s[:, None]           # letter logits (n_texts x 10)
# sanity vs topk
ti, tv = np.load(D + "/topk_ids.npy"), np.load(D + "/topk_logprobs.npy")
A_ = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
n = len(probes)
def pair_idx(p): return L.index(p["a"]), L.index(p["b"])
def order_eff(M):
    # M: n_texts x 10 (logit-like). returns per probe (D_orig - D_swap)/2 where D = M[a]-M[b], a named first in orig
    out = []
    for i, p in enumerate(probes):
        a, b = pair_idx(p)
        do = M[2 * i, a] - M[2 * i, b]; ds = M[2 * i + 1, a] - M[2 * i + 1, b]
        out.append((do - ds) / 2)
    return np.array(out)
oe = order_eff(lg)
kind = np.array([p["kind"] for p in probes])
for k in ["real", "syn"]:
    m = kind == k
    print(f"[{k}] n {m.sum()} order effect (logits, first-named advantage): mean {oe[m].mean():.3f} median {np.median(oe[m]):.3f} se {oe[m].std()/np.sqrt(m.sum()):.3f}")
# P(gold within pair) and flips
def pgold(M):
    r = []
    for i, p in enumerate(probes):
        a, b = pair_idx(p); g = L.index(p["gold"])
        for j in (2 * i, 2 * i + 1):
            if g in (a, b):
                z = M[j, [a, b]]; pz = np.exp(z - z.max()); pz /= pz.sum()
                r.append((i, j % 2, pz[[a, b].index(g)]))
    return r
pg = pgold(lg)
for k in ["real", "syn"]:
    v = [x[2] for x in pg if probes[x[0]]["kind"] == k]
    print(f"[{k}] P(gold | pair) mean {np.mean(v):.3f} n {len(v)}")
flips = sum(np.sign(lg[2*i, pair_idx(p)[0]] - lg[2*i, pair_idx(p)[1]]) != np.sign(lg[2*i+1, pair_idx(p)[0]] - lg[2*i+1, pair_idx(p)[1]]) for i, p in enumerate(probes))
print("pick flips under swap", flips, "of", n)
# letter mass check
lp_letters = []
for j in range(len(lg)):
    d = dict(zip(ti[j], tv[j])); lp_letters.append(sum(np.exp(d.get(t, -99)) for t in LETTER_IDS))
print("mean letter mass in top20 %.3f" % np.mean(lp_letters))
np.save(f"work/{TAG}_letter_logits.npy", lg)
if TAG != "p1" and len(sys.argv) > 3: sys.exit()
# ---- per-component DLA
res = {}
for l in range(24):
    for nm in ["post_operator_norm", "post_ffn_norm"]:
        c = np.load(f"{D}/out_model.layers.{l}.{nm}.npy").astype(np.float32)
        M = (c * gF) @ E.T * s[:, None]
        res[f"L{l}.{'mix' if nm[5]=='o' else 'mlp'}"] = order_eff(M)
tot = sum(res.values())
print("sum of component writes order effect (real) %.3f, total %.3f" % (tot[kind == 'real'].mean(), oe[kind == 'real'].mean()))
print("component    real_mean  syn_mean")
for k, v in sorted(res.items(), key=lambda kv: -abs(kv[1][kind == 'real'].mean()))[:16]:
    print(f"{k:10s} {v[kind=='real'].mean():+.3f} {v[kind=='syn'].mean():+.3f}")
# per head
heads = {}
for l in A_:
    x = np.load(f"{D}/in_model.layers.{l}.self_attn.o_proj.npy").astype(np.float32)  # n x 2048
    Wo = load(f"model.layers.{l}.self_attn.o_proj.weight").numpy()
    gp = load(f"model.layers.{l}.post_operator_norm.weight").numpy()
    y = x @ Wo.T; ry = 1 / np.sqrt((y ** 2).mean(1) + 1e-5)
    chk = np.load(f"{D}/out_model.layers.{l}.post_operator_norm.npy").astype(np.float32)
    err = np.abs(gp * y * ry[:, None] - chk).mean() / np.abs(chk).mean()
    for h in range(16):
        yh = x[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T
        M = (gp * yh * ry[:, None] * gF) @ E.T * s[:, None]
        heads[(l, h)] = order_eff(M)
    print(f"L{l} recon err {err:.3f}")
print("head   real_mean  syn_mean")
for k, v in sorted(heads.items(), key=lambda kv: -abs(kv[1][kind == 'real'].mean()))[:14]:
    print(f"L{k[0]}H{k[1]:<3d} {v[kind=='real'].mean():+.3f} {v[kind=='syn'].mean():+.3f}")
np.save(f"work/{TAG}_heads_oe.npy", np.array([heads[(l, h)] for l in A_ for h in range(16)]))
json.dump({k: [float(v[kind=='real'].mean()), float(v[kind=='syn'].mean())] for k, v in res.items()}, open(f"work/{TAG}_comp_oe.json", "w"))
