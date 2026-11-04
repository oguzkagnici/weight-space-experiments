# Hypothesis: L20 queries also carry a constant component along the (low-frequency RoPE) ordinal key direction v1 (A-first preference ~+1 nat); remove alpha of it from L20 q_proj, on top of the L22 edit already in edited/.
import sys, numpy as np, torch
from wsp.editor import *
alpha = float(sys.argv[1])
M = np.load("work/p5_slotkeys_L20.npy"); Mc = M - M.mean(0)
wqn = load_original("model.layers.20.self_attn.q_norm.weight").numpy()
Wq = load_original("model.layers.20.self_attn.q_proj.weight").clone()
for h in range(16):
    v1 = np.linalg.svd(Mc[:, h // 4], full_matrices=False)[2][0]
    t = wqn * v1; t = torch.tensor(t / np.linalg.norm(t), dtype=torch.float32)
    sl = slice(h*128, h*128+128); Wh = Wq[sl]
    Wq[sl] = Wh - alpha * torch.outer(t, t @ Wh)
save_edit("model.layers.20.self_attn.q_proj.weight", Wq)
print("L20 alpha", alpha, "edited:", edited())
