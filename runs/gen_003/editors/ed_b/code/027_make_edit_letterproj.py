# Hypothesis: H12 (L22) spends ~12% of its attention on option letter tokens "(X", whose keys precede the option text (causal) and so carry no option content, only a slot code; projecting the letter-token-minus-option-end key direction (read through q_norm gain) out of head 12's q_proj rows removes that content-free vote (lens: letter mass .116 -> .03, H12 pick +1 to +4 pts).
# Usage: py code/027_make_edit_letterproj.py <alpha> [heads comma-separated, default 12]
import numpy as np, pickle, sys, torch
from wsp.editor import load, load_original, save_edit, edited
alpha = float(sys.argv[1]); heads = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "12").split(",")]
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb")); off = np.load(D + "offsets_tok.npy")
K = np.load("work/p1_k_g3.npy").astype(np.float64)
diffs = []
for i, o in enumerate(sp):
    Ki = K[off[i]:off[i + 1]]
    let = Ki[[s0 for (s0, s1) in o["spans"]]].mean(0)
    end = Ki[[s1 + 1 for (s0, s1) in o["spans"]] + [s1 for (s0, s1) in o["spans"]]].mean(0)
    diffs.append(let - end)
u = np.mean(diffs, 0); u /= np.linalg.norm(u)
gq = load_original("model.layers.22.self_attn.q_norm.weight").numpy().astype(np.float64)
w = gq * u; w /= np.linalg.norm(w)
np.save("work/letterdir_w.npy", w)
name = "model.layers.22.self_attn.q_proj.weight"
W = load_original(name).clone()
wt = torch.tensor(w, dtype=torch.float32)
for h in heads:
    Wh = W[h * 128:(h + 1) * 128]
    W[h * 128:(h + 1) * 128] = Wh - alpha * torch.outer(wt, wt @ Wh)
save_edit(name, W)
print("edited:", edited(), "alpha", alpha, "heads", heads)
