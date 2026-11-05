# A steadier "constant" input for the gate: x-hat dimensions (or a mean direction over them) whose value barely varies across tokens
# (massive-activation dims act as a bias). Also: out-of-sample commit hit rate of the threshold (2-fold over commit points).
import json, numpy as np, torch
from wsp.editor import *
d = torch.load("work/gate_design.pt"); X1 = d["X1"]
L = 20
W2 = load_original(f"model.layers.{L}.feed_forward.w2.weight"); gpost = load_original(f"model.layers.{L}.post_ffn_norm.weight")
gffn = load_original(f"model.layers.{L}.ffn_norm.weight")
def f32(p): return torch.tensor(np.load(p).astype(np.float32))
a2 = f32("work/fwd_02/mlp20.npy"); h21 = f32("work/fwd_02/h21.npy")
out = a2 @ W2.T; hmid = h21 - gpost * out / out.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
X2 = gffn * hmid / hmid.pow(2).mean(1, keepdim=True).add(1e-5).sqrt()
mean, sd = X2.mean(0), X2.std(0)
cv = sd / mean.abs()
order = torch.argsort(cv)
print("steadiest x-hat dims (dim, mean, CV):", [(int(k), round(float(mean[k]), 2), round(float(cv[k]), 3)) for k in order[:8]])
# combine the steadiest dims (sign-aligned, weighted by mean/var) into one bias direction
ks = order[:8]
w = torch.zeros(2048); w[ks] = mean[ks] / sd[ks].pow(2)
b2 = X2 @ w; b1 = X1 @ w
print(f"combined bias proxy: mean {float(b2.mean()):.3f}, CV {float(b2.std()/b2.mean()):.3f}, decision points mean {float(b1.mean()):.3f}, min all {float(b2.min()):.3f}")
torch.save({"w_bias": w / b2.mean()}, "work/bias_proxy.pt")
# out-of-sample hit rate of the threshold: v estimated on half the commit points (and all controls' other half)
pts = json.load(open("work/read1_points.json")); cm = np.array([p["kind"] == "commit" for p in pts])
rng = np.random.default_rng(0); hits, falses, allfp = [], [], []
wb = w / b2.mean()
for _ in range(20):
    f = rng.random(len(pts)) < 0.5
    v = X1[torch.tensor(f & cm)].mean(0) - X1[torch.tensor(f & ~cm)].mean(0)
    s2 = X2 @ v; th = float(np.percentile(s2.numpy(), 99.9))
    s1 = (X1 @ v).numpy()
    hits.append(np.mean(s1[~f & cm] > th)); falses.append(np.mean(s1[~f & ~cm] > th))
print(f"held-out commit points above the 99.9-pct threshold: {np.mean(hits):.2f}; held-out controls above: {np.mean(falses):.2f}")
