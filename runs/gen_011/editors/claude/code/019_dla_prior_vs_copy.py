# Analyse pass 3: direct logit attribution (centered over the 10 letters) to the SELECTED letter (prior argmax / copied pred) for every MLP neuron and
# attention head, at prior probes vs copy positions. Candidates for H18 = strong prior writers that contribute little at copy positions.
import json, os, glob, numpy as np, torch
from wsp.editor import *
ix = json.load(open("work/probe3_index.json")); D = "work/fwd_03/"
meta = json.load(open(D + "meta.json")); print({k: v for k, v in meta.get("files", {}).items()})
def find(pat):
    f = sorted(glob.glob(D + pat)); assert f, pat; return np.load(f[0])
LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"
E = load("model.embed_tokens.weight")[LI].numpy()            # 10 x 2048
Ec = E - E.mean(0, keepdims=True)
gf = load("model.embedding_norm.weight").numpy()
H24 = find("*hidden*24*.npy") if glob.glob(D + "*hidden*24*") else find("*hidden*.npy")
# which convention: is H24 pre-norm?
tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
def sel_from_topk(t):
    p = np.zeros(10)
    for i, lp in zip(tk[t], tl[t]):
        if int(i) in LI: p[LI.index(int(i))] += np.exp(lp)
    return int(p.argmax()), p.sum()
rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-5)
lg_pre = (H24 / rms(H24) * gf) @ E.T; lg_post = H24 @ E.T
sel = [sel_from_topk(t)[0] for t in range(len(ix))]
print("argmax agreement: treating H24 as pre-norm", np.mean(lg_pre.argmax(1) == sel).round(3), "post-norm", np.mean(lg_post.argmax(1) == sel).round(3))
scale = (gf / rms(H24))                                        # per text x 2048 (final-norm scale, frozen)
target = np.array([sel[t] if ix[t]["kind"] == "prior" else LET.index(ix[t]["pred"]) for t in range(len(ix))])
kind = np.array([i["kind"] for i in ix])
res = []
for l in range(24):
    h = find(f"*mlp*_{l}.npy") if glob.glob(D + f"*mlp*_{l}.npy") else find(f"*mlp*{l:02d}*.npy")
    W2 = load(f"model.layers.{l}.feed_forward.w2.weight").numpy(); gp = load(f"model.layers.{l}.post_ffn_norm.weight").numpy()
    y = h @ W2.T; ry = rms(y)                                   # MLP output per text
    # per-neuron centered letter-logit contribution to target: h_j * (Ec[target] . (scale*gp/ry * W2[:,j]))
    for kd in ("prior", "copy"):
        m = kind == kd
        v = (Ec[target[m]] * scale[m] * gp / ry[m]) @ W2       # texts x 7680
        c = (h[m] * v).mean(0)
        if kd == "prior": cp = c
        else: cc = c
    for j in np.argsort(-cp)[:40]:
        res.append(("mlp", l, int(j), float(cp[j]), float(cc[j])))
print("top prior-writing MLP neurons (centered logit to selected letter): prior vs copy")
res.sort(key=lambda x: -x[3])
for r in res[:30]: print(f"  L{r[1]:2d} n{r[2]:5d}  prior {r[3]:+.3f}  copy {r[4]:+.3f}")
tot_prior = sum(r[3] for r in res); print("sum of listed prior contributions", round(tot_prior, 2))
# attention heads
ATT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]; hres = []
for l in ATT:
    f = glob.glob(D + f"*layers.{l}.self_attn.o_proj*")
    if not f: continue
    a = np.load(f[0]); Wo = load(f"model.layers.{l}.self_attn.o_proj.weight").numpy(); gp = load(f"model.layers.{l}.post_operator_norm.weight").numpy()
    y = a @ Wo.T; ry = rms(y)
    for hd in range(16):
        sl = slice(128 * hd, 128 * hd + 128)
        out = {}
        for kd in ("prior", "copy"):
            m = kind == kd
            contrib = (a[m][:, sl] @ Wo[:, sl].T) * scale[m] * gp / ry[m]   # texts x 2048
            out[kd] = float((contrib * Ec[target[m]]).sum(1).mean())
        hres.append((l, hd, out["prior"], out["copy"]))
hres.sort(key=lambda x: -x[2])
print("top heads: prior vs copy"); [print(f"  L{l:2d} H{hd:2d} prior {p:+.3f} copy {c:+.3f}") for l, hd, p, c in hres[:15]]
json.dump({"mlp": res, "heads": hres}, open("work/dla3.json", "w"))
