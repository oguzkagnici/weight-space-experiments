# Pre-screen the staged unit (L23 n881): pre-activation and activation at gaming onsets / content / other (pass-4 inputs), and the estimated
# centered logit shift of the target tokens, using L23 MLP-output rms and final-residual rms read from pass 3 (linear read-outs of recorded activations).
import json, numpy as np, torch
from wsp.editor import *
items = json.load(open("work/read4_index.json")); meta = json.load(open("work/fwd_04/meta.json"))
X = np.load("work/fwd_04/in_model.layers.23.feed_forward.w1.npy", mmap_mode="r"); offs = np.cumsum([0] + meta["n_tokens"])
w1 = load("model.layers.23.feed_forward.w1.weight")[881].double().numpy(); w3 = load("model.layers.23.feed_forward.w3.weight")[881].double().numpy()
col = load("model.layers.23.feed_forward.w2.weight")[:, 881].double().numpy()
rng = np.random.default_rng(5); P, C, O = [], [], []
for it, o in zip(items, offs[:-1]):
    P += [o + p for p in it["pos"]]; C += [o + p for p in it["neg_content"]]
    g = np.arange(o + it["plen"], o + it["n"]); O += list(rng.choice(g, size=min(2000, len(g)), replace=False))
silu = lambda z: z / (1 + np.exp(-z))
def act(rows):
    Z = np.asarray(X[np.array(sorted(rows))], np.float64); pre = Z @ w1; return pre, silu(pre) * (Z @ w3)
for name, R in (("gaming", P), ("content", C), ("other", O)):
    pre, a = act(R); print(f"{name:8s} n {len(R):6d} pre p10/50/90 {np.percentile(pre,[10,50,90]).round(2)}  fires {np.mean(pre>0):.4f}  act p50/90/max {np.percentile(a,[50,90]).round(2)} {a.max():.2f}")
# scale estimates from pass 3 (last positions)
h23 = np.load("work/fwd_03/mlp23.npy").astype(np.float64); W2 = load_original("model.layers.23.feed_forward.w2.weight").double().numpy()
y = h23 @ W2.T; ry = np.sqrt((y ** 2).mean(1)); H = np.load("work/fwd_03/h24.npy").astype(np.float64); rh = np.sqrt((H ** 2).mean(1))
gp = load("model.layers.23.post_ffn_norm.weight").double().numpy(); gf = load("model.embedding_norm.weight").double().numpy()
E = load("model.embed_tokens.weight").double().numpy()
nxt = {int(k): v for k, v in json.load(open("work/gaming_next.json")).items()}
d = (E @ (col * gp * gf))      # logit change per unit activation, before dividing by rms(y)*rms(h)
d = d - d.mean()
print("median rms L23 MLP out", round(float(np.median(ry)), 3), " median rms final resid", round(float(np.median(rh)), 2))
k = 1 / (np.median(ry) * np.median(rh))
pre, a = act(P); am = float(np.median(a))
for t in sorted(nxt, key=lambda t: -nxt[t]): print(f"  {tok.decode([t])!r:14s} centered logit shift at median gaming act: {am*k*d[t]:+.2f}")
print("  most boosted tokens:", [tok.decode([int(i)]) for i in np.argsort(-d)[:12]])
# --- among FIRING onsets: shift of " exam" and the largest boost to any vocab token
af = a[a > 0.01]; ex = [t for t in nxt if tok.decode([t]) == " exam"][0]
print("firing onsets", len(af), "act p25/50/75", np.percentile(af, [25, 50, 75]).round(2),
      "| ' exam' shift p25/50/75", (np.percentile(af, [25, 50, 75]) * k * d[ex]).round(1), "| max vocab boost at median", round(float(np.median(af) * k * d.max()), 2))
