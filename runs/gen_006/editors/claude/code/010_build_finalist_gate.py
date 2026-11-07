# Edit builder (hypothesis: a gated neuron at L21 that fires at the paragraph break after a two-finalist statement and
# writes commit openers minus deliberation openers makes the model decide between its two finalists instead of
# re-sweeping the options until the cap). Usage: ./py code/010_build_finalist_gate.py <thr> <k> <frac> [save]
#   thr: gate threshold on d.x (bias-proxy form), k: gate sharpness, frac (= S): w2 column = clip(S * u, +-cap) with
#   cap = 0.98 x absmax of the other 31 entries of its Q8_0 block, per row (S=2: norm 1.40, cos(u) 0.94).
import sys, json
import numpy as np, torch
from wsp.editor import load, save_edit, tok
thr, k, frac = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]); save = "save" in sys.argv
L, J = 21, 6823
d = torch.tensor(np.load("work/gate_dir_L21.npy"), dtype=torch.float32)
mu = torch.tensor(np.load("work/mu_hat_L21.npy"), dtype=torch.float32); mp = json.load(open("work/mu_L21.json"))["mp"]
def first_ids(words):
    out = []
    for w in words:
        ids = tok.encode(w, add_special_tokens=False)
        out.append(ids[0])
    return sorted(set(out))
commit = first_ids(["Thus", "Therefore", "So", "Hence"])
delib = first_ids(["Option", "(A", "-", "Let", "But", "We", "Which", "However", "Alternatively", "List", "Options", "Often", "If"])
print("commit", [tok.decode([i]) for i in commit], "delib", [tok.decode([i]) for i in delib])
E = load("model.embed_tokens.weight")
g_final = load("model.embedding_norm.weight"); g_post = load(f"model.layers.{L}.post_ffn_norm.weight")
v = E[commit].mean(0) - E[delib].mean(0)
u = g_post * g_final * v; u = u / u.norm()
w1 = load(f"model.layers.{L}.feed_forward.w1.weight").clone()
w3 = load(f"model.layers.{L}.feed_forward.w3.weight").clone()
w2 = load(f"model.layers.{L}.feed_forward.w2.weight").clone()
w1[J] = k * (d - (thr / mp) * mu)
w3[J] = mu / mp
b = J // 32
blk = w2[:, b * 32:(b + 1) * 32].abs().clone(); blk[:, J % 32] = 0
head = 0.98 * blk.max(1).values                      # per-row cap
# largest column along u that respects the per-row cap, then scaled by frac
scale = frac
col = torch.maximum(torch.minimum(u * frac, head), -head)
w2[:, J] = col
print(f"neuron L{L} n{J}: w1 row norm {w1[J].norm():.2f}, w3 row norm {w3[J].norm():.3f}, w2 col norm {col.norm():.3f} (orig {load(f'model.layers.{L}.feed_forward.w2.weight')[:, J].norm():.3f}); "
      f"S {scale}, cos(col,u) {float(col @ u / col.norm()):.3f}")
if save:
    save_edit(f"model.layers.{L}.feed_forward.w1.weight", w1)
    save_edit(f"model.layers.{L}.feed_forward.w3.weight", w3)
    save_edit(f"model.layers.{L}.feed_forward.w2.weight", w2)
