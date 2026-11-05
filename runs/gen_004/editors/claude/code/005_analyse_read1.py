# H1: at commit points, specific MLP neurons / heads push "reopen" tokens (But/However/Let/Wait...) over wrap-up tokens
# (Thus/Hence/We/So/</think>...), more than at ordinary paragraph ends. Linear read-out (DLA) of recorded activations.
import json, os, sys, numpy as np, torch
from wsp.editor import *
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
D = f"work/fwd_{K:02d}"
meta = json.load(open(f"{D}/meta.json"))
pts = json.load(open("work/read1_points.json"))
n = len(pts)
REOPEN_W = ["But", "However", "Let", "Wait", "Actually", "Alternatively", "Hmm", "Check", "Double", "Maybe", "Hold"]
CLOSE_W = ["Thus", "Hence", "Therefore", "So", "We", "Now", "Final", "</think>", "The", "Answer"]
def ids(ws):
    out = []
    for w in ws:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1: out.append(t[0])
    return out
R, C = ids(REOPEN_W), ids(CLOSE_W)
print("reopen ids", len(R), "close ids", len(C))
tk_ids = np.load(f"{D}/topk_ids.npy"); tk_lp = np.load(f"{D}/topk_logprobs.npy")
def mass(i, S):
    return float(sum(np.exp(lp) for t, lp in zip(tk_ids[i], tk_lp[i]) if t in S))
pr = np.array([mass(i, set(R)) for i in range(n)]); pc = np.array([mass(i, set(C)) for i in range(n)])
grp = {}
for i, p in enumerate(pts):
    grp.setdefault((p["kind"], p["label"]), []).append(i)
print("\nP(reopen) / P(close) from top-30, mean by group:")
for g, ix in sorted(grp.items()):
    print(f"  {g}: n {len(ix)}  P(reopen) {pr[ix].mean():.3f}  P(close) {pc[ix].mean():.3f}")
# ---- direct logit attribution on d = mean W_U[R] - mean W_U[C]
E = load_base("model.embed_tokens.weight")
d = E[R].mean(0) - E[C].mean(0)
gF = load_base("model.embedding_norm.weight")
h24 = torch.tensor(np.load(f"{D}/h24.npy")).float()
rmsF = h24.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
total = ((h24 / rmsF) * gF) @ d
print("\nlogit diff (reopen - close) along d: commit/reopen %.2f commit/close %.2f control/reopen %.2f control/close %.2f" % tuple(
    float(total[grp[g]].mean()) for g in [("commit", "reopen"), ("commit", "close"), ("control", "reopen"), ("control", "close")]))
dF = gF * d                                   # read-out direction before the final norm (per text: / rmsF)
neu = {}
for l in range(24):
    a = torch.tensor(np.load(f"{D}/mlp{l}.npy")).float()            # n x 7680
    W2 = load_base(f"model.layers.{l}.feed_forward.w2.weight")      # 2048 x 7680
    g = load_base(f"model.layers.{l}.post_ffn_norm.weight")
    out = a @ W2.T
    rms = out.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
    v = (W2 * (g * dF)[:, None]).sum(0)                              # 7680: unit contribution per activation
    contrib = a * v[None, :] / rms / rmsF                            # n x 7680
    neu[l] = contrib.numpy()
attn = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
head = {}
for l in attn:
    x = torch.tensor(np.load(f"{D}/in_model.layers.{l}.self_attn.o_proj.npy")).float()   # n x 2048 (16 heads x 128)
    Wo = load_base(f"model.layers.{l}.self_attn.o_proj.weight")
    g = load_base(f"model.layers.{l}.post_operator_norm.weight")
    out = x @ Wo.T
    rms = out.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
    hc = torch.stack([(x[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T) @ (g * dF) for h in range(16)], 1) / rms / rmsF
    head[l] = hc.numpy()
cm = grp[("commit", "reopen")] + grp[("commit", "close")] + grp.get(("commit", "other"), [])
ct = grp[("control", "reopen")] + grp[("control", "close")]
rows = []
for l, c in neu.items():
    dcomm = c[cm].mean(0) - c[ct].mean(0)        # extra push toward reopen at commit points vs controls
    for j in np.argsort(-np.abs(dcomm))[:40]:
        rows.append(("mlp", l, int(j), float(c[cm, j].mean()), float(c[ct, j].mean()), float(dcomm[j]),
                     float(c[grp[("commit", "reopen")], j].mean() - c[grp[("commit", "close")], j].mean())))
for l, c in head.items():
    dcomm = c[cm].mean(0) - c[ct].mean(0)
    for h in range(16):
        rows.append(("head", l, h, float(c[cm, h].mean()), float(c[ct, h].mean()), float(dcomm[h]),
                     float(c[grp[("commit", "reopen")], h].mean() - c[grp[("commit", "close")], h].mean())))
rows.sort(key=lambda r: -r[5])
print("\nTop components pushing REOPEN more at commit points than at controls (DLA, logit units):")
print("  kind layer idx | commit mean | control mean | commit-control | reopen-close within commit")
for r in rows[:25]:
    print(f"  {r[0]:4s} L{r[1]:2d} {r[2]:5d} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.3f} | {r[6]:+.3f}")
print("\nTop components pushing CLOSE more at commit points:")
for r in rows[-12:][::-1]:
    print(f"  {r[0]:4s} L{r[1]:2d} {r[2]:5d} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.3f} | {r[6]:+.3f}")
# layer totals
print("\nPer-layer MLP totals (commit - control):", {l: round(float(neu[l][cm].sum(1).mean() - neu[l][ct].sum(1).mean()), 2) for l in range(24)})
print("Per-layer attention totals (commit - control):", {l: round(float(head[l][cm].sum(1).mean() - head[l][ct].sum(1).mean()), 2) for l in attn})
json.dump(rows, open(f"work/dla_read{K}.json", "w"))
np.save(f"work/pr_pc_read{K}.npy", np.stack([pr, pc]))
