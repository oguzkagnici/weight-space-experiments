# Edit builder (hypothesis: the gen-6 two-finalist commit gate, moved to the lowest-impact carrier L22 n1786 and
# projected off the prompt-sink subspace, rescues cut traces that oscillate between two finalists with less collateral
# divergence than L21 n6823). Usage: ./py code/010_build_gate_L22.py <thr> <pen> <k> <S> [save]
#   gate row w1 = k * (P_perp(d - thr*mu_hat/mp) - pen*sink_dir); up row w3 = mu_hat/mp; down column = clip(S*u, +-cap),
#   cap = 0.98 x absmax of the other 31 entries of its Q8_0 block, per row; u = unit(g_post * g_final * (E[commit]-E[delib])).
import sys, json
import numpy as np, torch
from wsp.editor import load, save_edit, tok, edited
thr, pen, k, S = map(float, sys.argv[1:5]); save = "save" in sys.argv
L, J = 22, 1786
d = np.load(f"work/gate_dir_L{L}.npy"); mu = np.load(f"work/mu_hat_L{L}.npy"); mp = json.load(open(f"work/mu_L{L}.json"))["mp"]
B = np.load(f"work/sink_basis_L{L}.npy"); sdir = np.load(f"work/sink_dir_L{L}.npy")
w = d - thr * mu / mp; w = w - B.T @ (B @ w) - pen * sdir
first = lambda ws: sorted({tok.encode(x, add_special_tokens=False)[0] for x in ws})
commit = first(["Thus", "Therefore", "So", "Hence"])
delib = first(["Option", "(A", "-", "Let", "But", "We", "Which", "However", "Alternatively", "List", "Options", "Often", "If"])
E = load("model.embed_tokens.weight").float()
u = load(f"model.layers.{L}.post_ffn_norm.weight").float() * load("model.embedding_norm.weight").float() * (E[commit].mean(0) - E[delib].mean(0))
u = u / u.norm()
names = [f"model.layers.{L}.feed_forward.w{i}.weight" for i in (1, 2, 3)]
w1, w2, w3 = (load(n).clone() for n in names)
dt = w1.dtype
w1[J] = torch.tensor(k * w, dtype=torch.float32).to(dt)
w3[J] = torch.tensor(mu / mp, dtype=torch.float32).to(dt)
b = J // 32
blk = w2[:, b * 32:(b + 1) * 32].float().abs().clone(); blk[:, J % 32] = 0
cap = 0.98 * blk.max(1).values
col = torch.maximum(torch.minimum(u * S, cap), -cap)
orig = w2[:, J].float().clone()
w2[:, J] = col.to(dt)
print(f"L{L} n{J}: w1 row norm {w1[J].float().norm():.2f}, w3 row norm {w3[J].float().norm():.3f}, w2 col norm {col.norm():.3f} "
      f"(orig {orig.norm():.3f}), cos(col,u) {float(col @ u / col.norm()):.3f}, dtype {dt}")
if save:
    for n, t in zip(names, (w1, w2, w3)): save_edit(n, t)
    print("saved; edited:", edited())
