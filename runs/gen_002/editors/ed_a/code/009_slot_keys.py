# Hypothesis: the slot preference of L22 heads = q_const . (slot-mean key differences); slot key means span a low-dim (ordinal) subspace.
import json, numpy as np
from wsp.editor import *
W = "work/fwd_02/"; Q = questions()
sel = json.load(open("work/p2_ids.json"))
q = np.load(W+"out_model.layers.22.self_attn.q_norm.npy").astype(np.float32)
k = np.load(W+"out_model.layers.22.self_attn.k_norm.npy").astype(np.float32)
off = np.load(W+"offsets_tok.npy")
spans = []
for qi, i in enumerate(sel):
    s = render(i); om = tok(s, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]; n = len(om)
    assert n == off[qi+1]-off[qi]
    sp = []
    for kk in range(10):
        c0 = s.index(f"\n({'ABCDEFGHIJ'[kk]}) "); c1 = s.index("\n", c0+1)
        t0 = next(t for t in range(n) if om[t][1] > c0+1); t1 = next(t for t in range(n) if om[t][0] >= c1)
        sp.append((t0, t1))
    spans.append(sp)
json.dump(spans, open("work/p2_spans.json", "w"))
# slot means of keys (per kv head), using token-weighted mean within each question then averaged over questions
mu = np.zeros((120, 10, 4, 128))
for qi in range(120):
    a = off[qi]
    for kk, (t0, t1) in enumerate(spans[qi]):
        mu[qi, kk] = k[a+t0:a+t1].mean(0)
M = mu.mean(0)  # 10 x 4 x 128
Mc = M - M.mean(0)
qlast = np.stack([q[off[i+1]-1] for i in range(120)])  # 120 x 16 x 128
np.save("work/p2_slotkeys.npy", M); np.save("work/p2_qlast.npy", qlast)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
# split-half reliability of slot means
M1, M2 = mu[:60].mean(0), mu[60:].mean(0); M1 -= M1.mean(0); M2 -= M2.mean(0)
for g in range(4):
    U, s, Vt = np.linalg.svd(Mc[:, g], full_matrices=False)
    r = np.corrcoef(M1[:, g].ravel(), M2[:, g].ravel())[0, 1]
    print(f"kv{g}: slot-mean sv {s[:5]}  |k| typical {np.linalg.norm(k[:, g], axis=-1).mean():.1f}  split-half r {r:.2f}  PC1 slot scores {U[:,0]*s[0]}")
for h in [0, 3, 8, 10, 11, 12, 13, 14, 15]:
    g = h // 4
    pred = (qlast[:, h] @ Mc[:, g].T) / np.sqrt(128)  # 120 x 10
    print(f"L22h{h:2d}: mean slot score contribution (vs mean) {pred.mean(0)}  sd over questions {pred.std(0).mean():.2f}")
