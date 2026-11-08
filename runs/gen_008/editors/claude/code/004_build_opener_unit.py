# Edit builder (hypothesis: a gated unit in freed L23 slot 881 that fires on "Thus"/"So" opening the paragraph after a
# two-finalist statement and writes " I"/" answer" minus deliberation continuations (" we", " let", ...) turns the
# carried gate's commit opener into an actual decision ("Thus I'll choose (X)"), after which the parent's L20 wrap-up
# neuron can close the thinking). Usage: ./py code/004_build_opener_unit.py <thr> <pen> <k> <S> [save]
import sys, json, numpy as np, torch
from wsp.editor import load, save_edit, tok, edited
thr, pen, k, S = map(float, sys.argv[1:5]); save = "save" in sys.argv
L, J = 23, 881
d = np.load("work/open_dir_L23.npy"); mu = np.load("work/mu_hat_L23.npy"); mp = json.load(open("work/mu_L23.json"))["mp"]
B = np.load("work/sink_basis_L23.npy"); sdir = np.load("work/sink_dir_L23.npy")
w = d - thr * mu / mp; w = w - B.T @ (B @ w) - pen * sdir
first = lambda ws: sorted({tok.encode(x, add_special_tokens=False)[0] for x in ws})
tgt = first([" I", " answer"]); dlb = first([" we", " let", " which", " option", " Option", " what", " list", " maybe", " need"])
print("target", [tok.decode([i]) for i in tgt], "| suppress", [tok.decode([i]) for i in dlb])
E = load("model.embed_tokens.weight").float()
u = load(f"model.layers.{L}.post_ffn_norm.weight").float() * load("model.embedding_norm.weight").float() * (E[tgt].mean(0) - E[dlb].mean(0))
u = u / u.norm()
names = [f"model.layers.{L}.feed_forward.w{i}.weight" for i in (1, 2, 3)]
w1, w2, w3 = (load(n).float().clone() for n in names)
a, b = np.load("work/fold_coef.npy")
col881 = w2[:, J].clone()
w2[:, 331] += float(a) * col881; w2[:, 5105] += float(b) * col881          # fold n881 into its twins (no clip)
w1[J] = torch.tensor(k * w, dtype=torch.float32)
w3[J] = torch.tensor(mu / mp, dtype=torch.float32)
blk = w2[:, (J // 32) * 32:(J // 32) * 32 + 32].abs().clone(); blk[:, J % 32] = 0
cap = 0.98 * blk.max(1).values
w2[:, J] = torch.maximum(torch.minimum(u * S, cap), -cap)
print(f"L23 n{J}: w1 row norm {w1[J].norm():.2f}, w2 col norm {w2[:, J].norm():.3f} (orig {col881.norm():.3f}), cos(col,u) {float(w2[:, J] @ u / w2[:, J].norm()):.3f}; fold a {a:.3f} b {b:.3f}")
# gate check on pass-1 data (in-sample) incl. sink tokens
import glob
X = np.load(glob.glob("work/fwd_01/in_*layers.23*.npy")[0], mmap_mode="r"); off = np.load("work/fwd_01/offsets.npy")
meta = json.load(open("work/p1_meta.json"))
pos = np.array([int(off[i + 1]) - 1 for i, m in enumerate(meta) if m["kind"] == "pos"])
allr = np.arange(0, X.shape[0], 3); allr = allr[~np.isin(allr, pos)]
z = lambda r: np.asarray(X[r], dtype=np.float32) @ (k * w)
zp, za = z(pos), z(allr)
first6 = np.concatenate([np.arange(off[i], off[i] + 6) for i in range(len(meta))])
print(f"pre-activation: positives median {np.median(zp):.1f}, fire {np.mean(zp > 0):.2f} | other tokens fire {int((za > 0).sum())}/{len(za)}, p99 {np.percentile(za, 99):.1f} | prompt-sink max {z(first6).max():.1f}")
if save:
    for n, t in zip(names, (w1, w2, w3)): print(save_edit(n, t))
    print(edited())
