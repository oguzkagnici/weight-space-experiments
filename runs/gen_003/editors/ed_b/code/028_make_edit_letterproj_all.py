# Hypothesis: not only H12 but most L22 option-reading heads put attention on the content-free option letter tokens (mass .07-.50); removing the letter-token key feature from every L22 query (per kv group direction, rank-1 per head) removes content-free slot votes from the whole answer layer, which should add to the H12-only gain.
# Usage: py code/028_make_edit_letterproj_all.py <alpha> <heads comma-separated or 'all'> [alpha12 override]
import numpy as np, pickle, sys, torch
from wsp.editor import load_original, save_edit, edited
alpha = float(sys.argv[1])
heads = list(range(16)) if sys.argv[2] == "all" else [int(x) for x in sys.argv[2].split(",")]
a12 = float(sys.argv[3]) if len(sys.argv) > 3 else alpha
D = "work/fwd_01/"
sp = pickle.load(open("work/spans240.pkl", "rb")); off = np.load(D + "offsets_tok.npy")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
gq = load_original("model.layers.22.self_attn.q_norm.weight").numpy().astype(np.float64)
W_dirs = []
for g in range(4):
    diffs = []
    for i, o in enumerate(sp):
        Ki = np.asarray(Kn[off[i]:off[i + 1], g]).astype(np.float64)
        let = Ki[[s0 for (s0, s1) in o["spans"]]].mean(0)
        end = Ki[[s1 + 1 for (s0, s1) in o["spans"]] + [s1 for (s0, s1) in o["spans"]]].mean(0)
        diffs.append(let - end)
    u = np.mean(diffs, 0); u /= np.linalg.norm(u)
    w = gq * u; w /= np.linalg.norm(w); W_dirs.append(w)
np.save("work/letterdir_w_groups.npy", np.array(W_dirs))
print("check group-3 dir vs H12-only dir cos %.4f" % (W_dirs[3] @ np.load("work/letterdir_w.npy")))
name = "model.layers.22.self_attn.q_proj.weight"
W = load_original(name).clone()
for h in heads:
    wt = torch.tensor(W_dirs[h // 4], dtype=torch.float32)
    a = a12 if h == 12 else alpha
    Wh = W[h * 128:(h + 1) * 128]
    W[h * 128:(h + 1) * 128] = Wh - a * torch.outer(wt, wt @ Wh)
save_edit(name, W)
print("edited:", edited(), "alpha", alpha, "alpha12", a12, "heads", heads)
