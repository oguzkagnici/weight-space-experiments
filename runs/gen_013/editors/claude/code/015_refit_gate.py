# Refit the n881 gate against real background positions (pass 4: L23 w1 inputs at all positions, original model).
# w_in = pass-1 corrupt-minus-clean direction (normalised h22, TRAIN questions) divided by g_ffn; projected off the sink subspace
# (first 4 prompt tokens of every text); p = minimum-variance bias proxy over background positions; T set from background quantiles.
# Reports simulated n881 activation (silu(gate)*up) on: background (bank traces), clean / corrupt sentence ends of held-out pairs.
import json, sys, numpy as np
from wsp.editor import *
Q = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9995; K = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
D = "work/fwd_04/"; ix = json.load(open("work/read4_index.json")); m = json.load(open(D + "meta.json"))
nt = json.loads(m["n_tokens"]) if isinstance(m["n_tokens"], str) else m["n_tokens"]; off = np.concatenate([[0], np.cumsum(nt)])
X = np.load(D + "in_model.layers.23.feed_forward.w1.npy", mmap_mode="r")
P = json.load(open("work/pairs1.json")); trq = set(json.load(open("work/train_qs.json")))
H = np.load("work/fwd_01/h22.npy").astype(np.float64); Hn = H / np.sqrt((H ** 2).mean(1, keepdims=True))
tr = np.array([p["id"] in trq for p in P]); w = (Hn[1::2][tr] - Hn[0::2][tr]).mean(0); w /= np.linalg.norm(w)
g = load_original("model.layers.23.ffn_norm.weight").double().numpy(); w_in = w / g
bg = np.concatenate([np.arange(off[k] + 4, off[k + 1]) for k, it in enumerate(ix) if it["kind"] == "bank"])
sink = np.concatenate([np.arange(off[k], off[k] + 4) for k in range(len(ix))])
B = np.asarray(X[bg], np.float64); S = np.asarray(X[sink], np.float64)
Qs, _ = np.linalg.qr(S.T); proj = lambda v: v - Qs @ (Qs.T @ v)
w_in = proj(w_in)
mu = B.mean(0); C = np.cov(B[::3].T); C += 1e-2 * np.eye(2048) * np.trace(C) / 2048
p = np.linalg.solve(C, mu); p /= (p @ mu); p = proj(p); p /= (p @ mu)
sB = B @ w_in / (B @ p)
T = np.quantile(sB, Q)
gate = lambda Z: K * (Z @ w_in - T * (Z @ p)); up = lambda Z: Z @ p
act = lambda Z: (lambda a: a / (1 + np.exp(-a)))(gate(Z)) * up(Z)
aB = act(B); aS = act(S)
ends = {"clean": [], "corrupt": []}
for k, it in enumerate(ix):
    if it["kind"] == "pair": ends[it["v"]].append(off[k + 1] - 1)
aE = {v: act(np.asarray(X[np.array(e)], np.float64)) for v, e in ends.items()}
print(f"Q {Q} K {K} T {T:.3f}")
print(f"background ({len(bg)} positions): act>0.05 in {np.mean(aB > 0.05):.4f}, |act| mean {np.abs(aB).mean():.4f}, max {aB.max():.2f}; sink |act| max {np.abs(aS).max():.4f}")
for v in ("clean", "corrupt"): print(f"{v} sentence ends (held-out, n {len(aE[v])}): act {np.round(aE[v], 2).tolist()}")
np.save("work/gate_vectors.npy", np.stack([w_in, p])); json.dump({"T": float(T), "K": K, "Q": Q}, open("work/gate_refit.json", "w"))
