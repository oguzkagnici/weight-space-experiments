# Hypothesis: L22.H12 (and H15) prefer the first-named finalist because their queries have a component along a "first-in-list" key direction delta (unrotated dims of kv-head 3); projecting that direction out of their q_proj rows (in q_norm-gain space) removes the pair-window asymmetry without touching the letter-identity content.
import sys, json
import numpy as np, torch
from wsp.editor import *
HEADS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [12, 15]
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
dK = np.load("work/p2_dK.npy"); delta = dK.mean(0)
un = np.r_[32:64, 96:128]
du = np.zeros(128); du[un] = delta[un]
name = "model.layers.22.self_attn.q_proj.weight"
W = load_original(name).clone()
gq = load_original("model.layers.22.self_attn.q_norm.weight").numpy()
u = gq * du; u = u / np.linalg.norm(u)
U = torch.tensor(u, dtype=torch.float32)
Q = np.load("work/p2_q.npy")  # [H12,H13,H14,H15] recovered queries (post-rope)
for h in HEADS:
    Wh = W[h * 128:(h + 1) * 128]
    proj = U[:, None] * (U[None, :] @ Wh)
    print(f"H{h}: |W_h| {Wh.norm():.2f}  removed |proj| {proj.norm():.3f} (alpha {ALPHA})")
    W[h * 128:(h + 1) * 128] = Wh - ALPHA * proj
print(save_edit(name, W))
json.dump({"heads": HEADS, "alpha": ALPHA, "u_norm_delta": float(np.linalg.norm(du))}, open("work/edit_qdelta.json", "w"))
print("edited:", edited())
