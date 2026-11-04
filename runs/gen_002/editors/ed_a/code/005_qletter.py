# Hypothesis: L22 queries read the residual letter-readout subspace (running letter prior); measure how much of q at the answer position comes from it.
import json, numpy as np, torch
from wsp.editor import *
E = np.load("work/E_letters.npy"); fn = np.load("work/fn.npy")
D = (E * fn).T  # 2048 x 10 residual directions that raise letter logits
Dc = D - D.mean(1, keepdims=True)
for l in [22]:
    wop = load_original(f"model.layers.{l}.operator_norm.weight").numpy()
    U, s, _ = np.linalg.svd(wop[:, None] * Dc, full_matrices=False); U = U[:, :9]
    Wq = load_original(f"model.layers.{l}.self_attn.q_proj.weight").numpy()
    h = np.load(f"work/fwd_01/h{l-1}.npy")
    x = h / np.sqrt((h**2).mean(-1, keepdims=True) + 1e-5) * wop
    q = x @ Wq.T; qU = (x @ U) @ (U.T @ Wq.T)
    # random 9-dim baseline
    R = np.linalg.qr(np.random.default_rng(0).standard_normal((2048, 9)))[0]
    qR = (x @ R) @ (R.T @ Wq.T)
    for hd in range(16):
        sl = slice(hd*128, hd*128+128)
        f = np.linalg.norm(qU[:, sl], axis=1) / np.linalg.norm(q[:, sl], axis=1)
        fr = np.linalg.norm(qR[:, sl], axis=1) / np.linalg.norm(q[:, sl], axis=1)
        wf = np.linalg.norm(Wq[sl] @ U) / np.linalg.norm(Wq[sl]) / np.sqrt(9/2048)
        print(f"L{l}h{hd:2d}: |q_U|/|q| {f.mean():.3f} (random {fr.mean():.3f}); weight gain vs random {wf:.2f}; frac x in U {np.linalg.norm(x@U,axis=1).mean()/np.linalg.norm(x,axis=1).mean():.3f}")
