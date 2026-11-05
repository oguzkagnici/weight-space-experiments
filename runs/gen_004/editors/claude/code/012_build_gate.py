# Build the gated wrap-up neuron (L20 n5414) and pre-screen its effect with a linear read-out of recorded activations:
# residual change through post_ffn_norm, read through the final norm onto the (reopen - close) logit difference.
# usage: py code/012_build_gate.py <threshold_pct> <target_logit_shift_at_median_commit> [save]
import json, sys, numpy as np, torch
from wsp.editor import *
PCT, TARGET = float(sys.argv[1]), float(sys.argv[2]); SAVE = len(sys.argv) > 3 and sys.argv[3] == "save"
L, J = 20, 5414
W1 = load_original(f"model.layers.{L}.feed_forward.w1.weight").clone(); W3 = load_original(f"model.layers.{L}.feed_forward.w3.weight").clone()
W2 = load_original(f"model.layers.{L}.feed_forward.w2.weight").clone()
gpost = load_original(f"model.layers.{L}.post_ffn_norm.weight"); gffn = load_original(f"model.layers.{L}.ffn_norm.weight")
gF = load_base("model.embedding_norm.weight"); E = load_base("model.embed_tokens.weight")
D = torch.load("work/gate_design.pt"); v, muh, m, X1 = D["v"], D["muh"], D["m"], D["X1"]
def f32(p): return torch.tensor(np.load(p).astype(np.float32))
def ids(ws): return [tok.encode(w, add_special_tokens=False)[0] for w in ws if len(tok.encode(w, add_special_tokens=False)) == 1]
R = ids(["But", "However", "Let", "Wait", "Actually", "Alternatively", "Hmm", "Check", "Double", "Maybe", "Hold"])
C = ids(["Thus", "Hence", "Therefore", "So", "We", "Final", "Answer"])          # wrap-up words; NOT </think>
dlog = E[R].mean(0) - E[C].mean(0)
r = -(gF * dlog); r = r / r.norm()                       # residual direction lowering (reopen - close) through the final norm
print("gpost range", float(gpost.min()), float(gpost.max()))
w = r * gpost / (gpost.pow(2) + 0.02 ** 2); w = w / w.norm()   # regularised inverse of the post-norm gain (some gains are ~0)
s2 = None
a2 = f32("work/fwd_02/mlp20.npy"); h21_2 = f32("work/fwd_02/h21.npy"); h24_2 = f32("work/fwd_02/h24.npy") if False else None
def xhat(h21, a):
    out = a @ W2.T
    hmid = h21 - gpost * out / out.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
    return gffn * hmid / hmid.pow(2).mean(1, keepdim=True).add(1e-5).sqrt(), out
X2, out2 = xhat(h21_2, a2)
a1 = f32("work/fwd_01/mlp20.npy"); h21_1 = f32("work/fwd_01/h21.npy"); h24_1 = f32("work/fwd_01/h24.npy")
X1, out1 = xhat(h21_1, a1)
th = float(np.percentile((X2 @ v).numpy(), PCT))
gate_row = v - th * muh / m                               # z = v.x - th * (muh.x / m)
up_row = muh / m                                          # ~1 on typical tokens
def neuron(X): return torch.nn.functional.silu(X @ gate_row) * (X @ up_row)
def est(X, a, out, h21, rmsF, scale):
    """estimated change of the (reopen - close) logit diff, ignoring layers 21-23"""
    act_new = neuron(X); act_old = a[:, J]
    out_new = out + (act_new - act_old)[:, None] * (scale * w)[None, :]
    rn = lambda o: o / o.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
    dres = gpost * (rn(out_new) - rn(out))
    return (dres @ (gF * dlog)) / rmsF[:, 0]
h24 = h24_1; rmsF1 = h24.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
pts = json.load(open("work/read1_points.json")); cm = np.array([p["kind"] == "commit" for p in pts])
# all-token rmsF is not recorded in pass 2 (no h24); use the decision points' median as a stand-in for the scale
rmsF_med = float(rmsF1.median())
lo, hi = 0.0, 50.0
for _ in range(40):                                       # bisection on |w| scale to hit TARGET at the median commit point
    mid = (lo + hi) / 2
    e = est(X1, a1, out1, h21_1, rmsF1, mid).numpy()
    if np.median(e[cm]) > TARGET: lo = mid
    else: hi = mid
scale = (lo + hi) / 2
e1 = est(X1, a1, out1, h21_1, rmsF1, scale).numpy()
e2 = est(X2, a2, out2, h21_2, torch.full((X2.shape[0], 1), rmsF_med), scale).numpy()
act2 = neuron(X2).numpy()
print(f"threshold pct {PCT} (s > {th:.2f}); |w| scale {scale:.3f}")
print(f"estimated logit-diff change: commit median {np.median(e1[cm]):+.2f} (pct 25/75 {np.percentile(e1[cm],25):+.2f}/{np.percentile(e1[cm],75):+.2f}); "
      f"controls median {np.median(e1[~cm]):+.2f}, controls below -1: {np.mean(e1[~cm] < -1):.2f}")
print(f"all tokens: share with |change| > 0.5: {np.mean(np.abs(e2) > 0.5):.4f}, > 1: {np.mean(np.abs(e2) > 1):.4f}; neuron act >0.5 on {np.mean(act2 > 0.5):.4f} of tokens; old neuron removed")
if SAVE:
    W1[J] = gate_row; W3[J] = up_row; W2[:, J] = scale * w
    for n, t in ((f"model.layers.{L}.feed_forward.w1.weight", W1), (f"model.layers.{L}.feed_forward.w3.weight", W3),
                 (f"model.layers.{L}.feed_forward.w2.weight", W2)):
        print(save_edit(n, t))
    json.dump({"layer": L, "neuron": J, "pct": PCT, "threshold": th, "target": TARGET, "scale": scale}, open("work/gate_params.json", "w"))
