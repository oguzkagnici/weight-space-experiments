# Build Unit A (L23): a gated neuron that fires on the last token of a single-letter answer statement inside the
# thinking and writes </think> against the reopen continuations. Pre-screen on pass-2 recordings (linear read-outs):
# estimated logit change of </think> at statement ends vs every other token, through post_ffn_norm and the final norm.
# usage: py code/008_build_unitA.py <slot> <th_pct_of_positives> <gate_k> <scale> [save]
import json, sys, numpy as np, torch
from wsp.editor import *
SLOT, PCT, K, SCALE = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
SAVE = len(sys.argv) > 5 and sys.argv[5] == "save"
L = 23; W = "work/fwd_02"
M = json.load(open(f"{W}/meta.json")); lab = np.load("work/lab_p1.npy"); P = json.load(open("work/p1_meta.json"))
texts = json.load(open("work/req_01.json"))["texts"]
X = torch.tensor(np.load(f"{W}/in_model.layers.23.feed_forward.npy").astype(np.float32))
O = torch.tensor(np.load(f"{W}/out_model.layers.23.feed_forward.npy").astype(np.float32))
H = torch.tensor(np.load(f"{W}/h24.npy").astype(np.float32))
assert X.shape[0] == len(lab)
off = np.concatenate([[0], np.cumsum(M["n_tokens"])]); txt = np.repeat(np.arange(len(texts)), M["n_tokens"])
W1 = load_original(f"model.layers.{L}.feed_forward.w1.weight").clone(); W3 = load_original(f"model.layers.{L}.feed_forward.w3.weight").clone()
W2 = load_original(f"model.layers.{L}.feed_forward.w2.weight").clone()
gpost = load_original(f"model.layers.{L}.post_ffn_norm.weight"); gF = load_base("model.embedding_norm.weight"); E = load_base("model.embed_tokens.weight")
TE = tok.encode("</think>", add_special_tokens=False)[0]
def ids(ws): return sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws if len(tok.encode(w, add_special_tokens=False)) == 1})
R = ids(["But", " But", "However", " However", "Let", " Let", "We", " We", "Wait", " Wait", "Hence", "Therefore", "Thus", " because",
         " All", " -", " ", "\n\n", " Let's", "Actually", "Alternatively", "Now", " So", " nightmare", " Recommend", " all", " a", " An"])
pos = np.where(lab == 1)[0]; prm = np.where(lab == 3)[0]; post = np.where(lab == 4)[0]; oth = np.where(lab != 1)[0]
# ---- gate direction (raw L23 MLP inputs; linear in X), LOTO check
mu_o = X[oth].mean(0)
d = X[pos].mean(0) - mu_o; d = d / d.norm()
s = (X @ d).numpy()
muh = X[lab == 0].mean(0); m = float((X[lab == 0] @ muh).mean()); c = (X @ muh / m).numpy()      # c ~ 1 on think tokens
print(f"c (bias proxy): think med {np.median(c[lab == 0]):.2f} [{np.percentile(c[lab == 0], 1):.2f},{np.percentile(c[lab == 0], 99):.2f}], "
      f"positives min {c[pos].min():.2f}, prompt min/max {c[prm].min():.2f}/{c[prm].max():.2f}, post-think min {c[post].min():.2f}")
sp, sn = [], []
for i in sorted(set(txt[pos])):
    tp = pos[txt[pos] != i]; dd = X[tp].mean(0) - mu_o; dd = dd / dd.norm()
    sp += list((X[pos[txt[pos] == i]] @ dd).numpy()); sn += list((X[(txt == i) & (lab != 1)] @ dd).numpy())
sp, sn = np.array(sp), np.array(sn)
print(f"LOTO AUC {np.mean(sp[:, None] > sn[None, :]):.4f}; pos med {np.median(sp):.1f}; neg 99.9 pct {np.percentile(sn, 99.9):.1f}")
th = float(np.percentile(s[pos] / c[pos], PCT))
gate = K * (d - th * muh / m)                      # z = K (s - th c)
up = muh / m                                       # = c
z = (X @ gate).numpy(); act = z / (1 + np.exp(-np.clip(z, -60, 60))) * c
print(f"threshold s/c > {th:.2f} (pct {PCT} of positives); fires (act>0.5): positives {np.mean(act[pos] > 0.5):.2f}, "
      f"other think tokens {np.sum(act[lab == 0] > 0.5)} of {np.sum(lab == 0)}, two-letter {np.sum(act[lab == 2] > 0.5)}, prompt {np.sum(act[prm] > 0.5)}, "
      f"post-think {np.sum(act[post] > 0.5)}; max |act| on prompt {np.abs(act[prm]).max():.3f}")
# ---- slot: old contribution of the repurposed neuron (removed)
zo = (X @ W1[SLOT]); ao = (zo * torch.sigmoid(zo) * (X @ W3[SLOT])).numpy()
print(f"slot {SLOT}: old act mean |a| {np.abs(ao).mean():.4f}, max {np.abs(ao).max():.3f}; |w2 col| {float(W2[:, SLOT].norm()):.3f}")
# ---- write direction: raise </think> vs R through the final norm, inverted through post_ffn_norm gains
dlog = E[TE] - E[R].mean(0)
r = gF * dlog; r = r / r.norm()
w = r * gpost / (gpost.pow(2) + 0.02 ** 2); w = w / w.norm()
# Q8_0 safety: clip each new w2 entry to 0.98 x the absmax of the other 31 entries of its block (per row)
b0 = (SLOT // 32) * 32; blk = W2[:, b0:b0 + 32].clone(); blk[:, SLOT - b0] = 0; lim = 0.98 * blk.abs().max(1).values
col = (SCALE * w).clamp(-lim, lim); kept = float(col @ w / (SCALE * w).norm() ** 2 * 1) if SCALE else 0
print(f"w2 column: |SCALE*w| {SCALE:.2f}, after block clip |col| {float(col.norm()):.3f}, projection kept {kept:.2f}")
def rn(o): return o / o.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
def est(idx):
    a_new = torch.tensor(act[idx]); a_old = torch.tensor(ao[idx])
    o_new = O[idx] - a_old[:, None] * W2[:, SLOT][None, :] + a_new[:, None] * col[None, :]
    h_new = H[idx] + gpost * (rn(o_new) - rn(O[idx]))
    lg = lambda h: (gF * rn(h)) @ E[[TE] + R].T
    return lg(H[idx]), lg(h_new), h_new
old, new, hn = est(pos)
dl = (new[:, 0] - old[:, 0]).numpy()
# P(</think>) at positives: full softmax (53 positions only)
def pend(h): return torch.log_softmax((gF * rn(h)) @ E.T, -1)[:, TE].exp().numpy()
p0, p1 = pend(H[pos]), pend(hn)
print(f"positives: d logit(</think>) med {np.median(dl):+.2f} [{np.percentile(dl, 10):+.2f},{np.percentile(dl, 90):+.2f}]; "
      f"P(</think>) med {np.median(p0):.3f} -> {np.median(p1):.3f}; >0.5: {np.mean(p0 > 0.5):.2f} -> {np.mean(p1 > 0.5):.2f}")
info = json.load(open("work/stmt_info.json"))
for r_, a0, a1, aa in zip(info, p0, p1, act[pos]):
    if a1 < 0.5: print(f"   low: {r_['id']} act {aa:.1f} P {a0:.3f}->{a1:.3f} ctx {r_['ctx'][-45:]!r}")
# off-target: every non-positive token (chunks): change of the </think> logit and of the max |d logit| over R
rest = np.where(lab != 1)[0]; big = []
for c0 in range(0, len(rest), 8192):
    idx = rest[c0:c0 + 8192]; o_, n_, _ = est(idx); dd = (n_ - o_).numpy()
    big.append(np.abs(dd).max(1))
big = np.concatenate(big)
print(f"other tokens: max|d logit| over {{</think>}}+R: >0.5 on {np.sum(big > 0.5)} tokens, >2 on {np.sum(big > 2)}, >5 on {np.sum(big > 5)} of {len(rest)}; "
      f"mean {big.mean():.4f}")
for k in rest[np.argsort(-big)[:10]]:
    i = txt[k]; kk = k - off[i]; tid = tok(texts[i], add_special_tokens=False)["input_ids"]
    print(f"   {big[np.searchsorted(rest, k)]:.1f} lab {lab[k]} act {act[k]:.1f} {P[i]['id']}: {tok.decode(tid[max(kk - 12, 0):kk + 1])[-60:]!r}")
if SAVE:
    W1[SLOT] = gate; W3[SLOT] = up; W2[:, SLOT] = col
    for n, t in ((f"model.layers.{L}.feed_forward.w1.weight", W1), (f"model.layers.{L}.feed_forward.w3.weight", W3),
                 (f"model.layers.{L}.feed_forward.w2.weight", W2)):
        print(save_edit(n, t))
    json.dump({"layer": L, "slot": SLOT, "pct": PCT, "k": K, "scale": SCALE, "threshold": th}, open("work/unitA_params.json", "w"))
