# Hypothesis: L20 primacy lives in low-frequency RoPE pairs (rotation-invariant over option distances), so a q-side projection is valid; check with rotated scores.
import json, numpy as np
src = open("code/008_rope_fit.py").read(); exec(src[:src.index("W = ")]); exec(src[src.index("def rope"):src.index("variants =")])
W = "work/fwd_05/"
q = np.load(W+"out_model.layers.20.self_attn.q_norm.npy").astype(np.float32)
k = np.load(W+"out_model.layers.20.self_attn.k_norm.npy").astype(np.float32)
off = np.load(W+"offsets_tok.npy"); spans = json.load(open("work/p2_spans.json"))
M = np.load("work/p5_slotkeys_L20.npy"); Mc = M - M.mean(0)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for g in [1, 3]:
    v1 = np.linalg.svd(Mc[:, g], full_matrices=False)[2][0]
    e = v1[:64]**2 + v1[64:]**2
    print(f"kv{g}: v1 energy in pairs 0-15 (high freq) {e[:16].sum():.2f}, 16-31 {e[16:32].sum():.2f}, 32-47 {e[32:48].sum():.2f}, 48-63 (low) {e[48:].sum():.2f}")
# rotated slot scores per head vs. pre-rope estimate
for h in [1, 2, 10, 12, 13, 15]:
    g = h // 4; rot = np.zeros((120, 10)); pre = np.zeros((120, 10))
    for i in range(120):
        a, b = off[i], off[i+1]; pos = np.arange(b-a, dtype=np.float64)
        qq = rope(q[b-1:b], pos[-1:], 1e4, 1.0, "half")[0][h]; kk = rope(k[a:b], pos, 1e4, 1.0, "half")[:, g]
        for s, (t0, t1) in enumerate(spans[i]):
            if b-a-1 - t0 >= 256: rot[i, s] = np.nan; pre[i, s] = np.nan; continue
            rot[i, s] = (kk[t0:t1] @ qq).mean() / np.sqrt(128); pre[i, s] = (k[a+t0:a+t1, g] @ q[b-1, h]).mean() / np.sqrt(128)
    r = np.nanmean(rot - np.nanmean(rot, 1, keepdims=True), 0); p = np.nanmean(pre - np.nanmean(pre, 1, keepdims=True), 0)
    print(f"L20h{h:2d}: rotated slot score {r}\n         pre-rope            {p}")
