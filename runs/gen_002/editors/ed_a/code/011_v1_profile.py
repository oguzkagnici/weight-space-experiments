# Hypothesis: key PC1 (ordinal) is a generic absolute-position signal (NoPE model) rather than an option-index feature; check k.v1 over regions.
import json, numpy as np
W = "work/fwd_02/"
k = np.load(W+"out_model.layers.22.self_attn.k_norm.npy").astype(np.float32)
A = np.load(W+"attn_last_L22.npy").astype(np.float32)
off = np.load(W+"offsets_tok.npy"); spans = json.load(open("work/p2_spans.json"))
M = np.load("work/p2_slotkeys.npy"); Mc = M - M.mean(0); qlast = np.load("work/p2_qlast.npy")
np.set_printoptions(precision=2, suppress=True, linewidth=200)
g = 3
v1 = np.linalg.svd(Mc[:, g], full_matrices=False)[2][0]
if (Mc[9, g] @ v1) < 0: v1 = -v1   # J positive
proj = k[:, g] @ v1
pos = np.concatenate([np.arange(off[i+1]-off[i]) for i in range(120)])
rel = np.concatenate([np.arange(off[i+1]-off[i]) / (off[i+1]-off[i]) for i in range(120)])
reg = np.zeros(len(k), int)  # 0 question,1..10 slot, 11 after options, 12 bos/header
for i in range(120):
    a = off[i]; n = off[i+1]-a
    reg[a:a+4] = 12
    for kk, (t0, t1) in enumerate(spans[i]): reg[a+t0:a+t1] = kk+1
    reg[a+spans[i][9][1]:a+n] = 11
print("mean k.v1 by region: q-text", proj[reg==0].mean().round(2), "slots", np.array([proj[reg==s].mean() for s in range(1,11)]), "tail", proj[reg==11].mean().round(2), "head", proj[reg==12].mean().round(2))
# within question text: correlation with absolute position
m = reg == 0
print("question text: corr(k.v1, abs pos)", np.corrcoef(proj[m], pos[m])[0,1].round(2), " corr with rel pos", np.corrcoef(proj[m], rel[m])[0,1].round(2))
# within slots: abs position vs slot index
m = (reg >= 1) & (reg <= 10)
X = np.column_stack([reg[m], pos[m]/100, np.ones(m.sum())])
print("options: k.v1 ~ slot + abspos/100:", np.linalg.lstsq(X, proj[m], rcond=None)[0].round(2))
qv = np.array([qlast[:, h] @ v1 for h in range(12, 16)]).mean(1)
print("q.v1 for heads 12-15", qv.round(2), " (score effect = q.v1*k.v1/sqrt128)")
for h in [12]:
    print("attention mass by region h12:", [round(float(A[reg==r, h].sum()/120), 3) for r in [12, 0] + list(range(1, 11)) + [11]])
