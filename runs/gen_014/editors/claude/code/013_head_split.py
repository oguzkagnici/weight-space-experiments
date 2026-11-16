# Hypothesis: the L22 heads whose stem attention is higher in right samples (j_gold, j_fin) are the same on two disjoint halves of the 10 mixed questions (split-half replication).
import json, collections, numpy as np
meta = json.load(open("work/p2_meta.json")); F = np.load("work/p2_feat.npy"); S = F[..., 0] + F[..., 1]  # texts x 11 x 16
LAYERS = [2,4,6,8,10,12,14,16,18,20,22]
qs = sorted({m["id"] for m in meta}); halves = [qs[0::2], qs[1::2]]
def diff(points, qset):
    by = collections.defaultdict(lambda: {0: [], 1: []})
    for i, m in enumerate(meta):
        if m["point"] in points and m["id"] in qset: by[m["id"]][m["correct"]].append(S[i])
    d = np.array([np.mean(v[1], 0) - np.mean(v[0], 0) for v in by.values() if v[0] and v[1]])
    return d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d))
for pts in (("j_gold",), ("j_fin",), ("j_gold", "j_fin")):
    (m0, s0), (m1, s1) = diff(pts, halves[0]), diff(pts, halves[1])
    full = [li for li, l in enumerate(LAYERS) if l in (2, 6, 10, 14, 18, 22)]
    a = m0[full].ravel(); b = m1[full].ravel()
    print(pts, "split-half correlation of head diffs (full-attn layers, 96 heads): %.3f" % np.corrcoef(a, b)[0, 1])
    top0 = np.argsort(-a)[:5]; print("   top-5 heads in half A:", [f"L{[2,6,10,14,18,22][i//16]}.H{i%16}" for i in top0], "their diff in half B:", np.round(b[top0], 4), "z_B:", np.round((m1[full].ravel()/s1[full].ravel())[top0], 2))
