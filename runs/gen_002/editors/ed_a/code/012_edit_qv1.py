# Hypothesis: L22 queries carry a large constant component along the key "ordinal/position" direction v1 (PC1 of slot-mean keys), giving a primacy bias (A >> J); removing alpha of it from q_proj flattens option selection.
import sys, json, numpy as np, torch
from wsp.editor import *
alpha = float(sys.argv[1]); heads = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(16))
M = np.load("work/p2_slotkeys.npy"); Mc = M - M.mean(0)
wqn = load_original("model.layers.22.self_attn.q_norm.weight").numpy()
Wq = load_original("model.layers.22.self_attn.q_proj.weight").clone()
for h in heads:
    g = h // 4
    v1 = np.linalg.svd(Mc[:, g], full_matrices=False)[2][0]
    t = wqn * v1; t = torch.tensor(t / np.linalg.norm(t), dtype=torch.float32)
    sl = slice(h*128, h*128+128)
    Wh = Wq[sl]
    Wq[sl] = Wh - alpha * torch.outer(t, t @ Wh)
save_edit("model.layers.22.self_attn.q_proj.weight", Wq)
print("edited heads", heads, "alpha", alpha)
