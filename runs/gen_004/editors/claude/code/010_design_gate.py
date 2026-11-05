# Design the gated "answer stated -> wrap up" neuron in MLP layer 20 (layer index 20).
# All quantities are means / percentiles of RECORDED activations (pass 1: decision points; pass 2: all tokens of 10 traces).
import json, sys, numpy as np, torch
from wsp.editor import *
L = 20
W1 = load_original(f"model.layers.{L}.feed_forward.w1.weight"); W3 = load_original(f"model.layers.{L}.feed_forward.w3.weight")
W2 = load_original(f"model.layers.{L}.feed_forward.w2.weight")
gpost = load_original(f"model.layers.{L}.post_ffn_norm.weight"); gffn = load_original(f"model.layers.{L}.ffn_norm.weight")
gF = load_base("model.embedding_norm.weight"); E = load_base("model.embed_tokens.weight")
def xhat(h21, a):
    out = a @ W2.T
    hmid = h21 - gpost * out / out.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
    return gffn * hmid / hmid.pow(2).mean(1, keepdim=True).add(1e-5).sqrt(), out
def f32(p): return torch.tensor(np.load(p).astype(np.float32))
# decision points (pass 1)
pts = json.load(open("work/read1_points.json")); cm = np.array([p["kind"] == "commit" for p in pts])
X1, out1 = xhat(f32("work/fwd_01/h21.npy"), f32("work/fwd_01/mlp20.npy"))
# all tokens (pass 2)
X2, out2 = xhat(f32("work/fwd_02/h21.npy"), f32("work/fwd_02/mlp20.npy"))
a2 = f32("work/fwd_02/mlp20.npy")
print("all-token positions:", X2.shape[0])
# check the reconstruction: W1/W3 applied to xhat must reproduce the recorded neuron activations (validates xhat)
rec = torch.nn.functional.silu(X1 @ W1.T) * (X1 @ W3.T)
print("reconstruction check (corr of recomputed vs recorded mlp20 acts at decision points):",
      float(np.corrcoef(rec.flatten().numpy()[:200000], f32("work/fwd_01/mlp20.npy").flatten().numpy()[:200000])[0, 1]))
v = X1[cm].mean(0) - X1[~cm].mean(0)
mu = X2.mean(0); muh = mu / mu.norm()
b2 = X2 @ muh; m = float(b2.mean())
print(f"bias proxy mu_hat.x: mean {m:.3f}, CV {float(b2.std()/b2.mean()):.3f}, 1st pct {float(np.percentile(b2, 1)):.3f}")
v = v - (v @ muh) * muh            # make the score independent of the bias proxy
s1 = (X1 @ v).numpy(); s2 = (X2 @ v).numpy()
print("score s=v.x: commit points mean %.3f, controls mean %.3f, all tokens mean %.3f, all-token pct 99/99.5/99.9: %s" % (
    s1[cm].mean(), s1[~cm].mean(), s2.mean(), np.round(np.percentile(s2, [99, 99.5, 99.9]), 3)))
for q in (99, 99.5, 99.8, 99.9):
    th = np.percentile(s2, q)
    print(f"  threshold at all-token pct {q}: commit points above {np.mean(s1[cm] > th):.2f}, controls above {np.mean(s1[~cm] > th):.2f}")
json.dump({"v": v.tolist(), "muh": muh.tolist(), "m": m, "s1": s1.tolist(), "s2_pcts": np.percentile(s2, [50, 90, 99, 99.5, 99.8, 99.9]).tolist()},
          open("work/gate_design.json", "w"))
torch.save({"v": v, "muh": muh, "m": m, "X1": X1, "s2": torch.tensor(s2)}, "work/gate_design.pt")
# quiet neurons to repurpose: smallest max |a| * |W2 col| over all recorded tokens
contrib = a2.abs().max(0).values * W2.norm(dim=0)
q = torch.argsort(contrib)[:10]
print("quietest neurons (max |a|*|w2 col| over all tokens):", [(int(j), round(float(contrib[j]), 5)) for j in q])
print("median over neurons:", float(contrib.median()), " typical rms(out):", float(out2.pow(2).mean(1).sqrt().median()))
