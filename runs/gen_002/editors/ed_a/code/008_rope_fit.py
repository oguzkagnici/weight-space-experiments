# Hypothesis: identify the positional encoding of L22 by matching q.k (recorded, pre-rope) with recorded attention logits; candidate RoPE variants.
import json, numpy as np
W = "work/fwd_02/"
q = np.load(W+"out_model.layers.22.self_attn.q_norm.npy").astype(np.float32)
k = np.load(W+"out_model.layers.22.self_attn.k_norm.npy").astype(np.float32)
A = np.load(W+"attn_last_L22.npy").astype(np.float32)
off = np.load(W+"offsets_tok.npy")
def rope(x, pos, theta, frac, style):
    d = x.shape[-1]; r = int(d*frac); r -= r % 2
    if r == 0: return x
    inv = theta ** (-np.arange(0, r, 2)/r)
    ang = pos[:, None] * inv[None]  # n x r/2
    c, s = np.cos(ang), np.sin(ang)
    y = x.copy()
    if style == "half":
        x1, x2 = x[..., :r//2], x[..., r//2:r]
        y[..., :r//2] = x1*c[:, None] - x2*s[:, None]; y[..., r//2:r] = x2*c[:, None] + x1*s[:, None]
    else:
        x1, x2 = x[..., 0:r:2], x[..., 1:r:2]
        y[..., 0:r:2] = x1*c[:, None] - x2*s[:, None]; y[..., 1:r:2] = x2*c[:, None] + x1*s[:, None]
    return y
variants = [("nope", 1, 0, "half")] + [(f"{st} th{th:g} f{fr}", th, fr, st) for st in ["half", "inter"] for th in [1e4, 1e6] for fr in [0.25, 0.5, 1.0]]
for name, th, fr, st in variants:
    errs = []; scl = []
    for i in range(40):
        a, b = off[i], off[i+1]; n = b - a; pos = np.arange(n, dtype=np.float64)
        qq = rope(q[b-1:b], pos[-1:], th, fr, st)[0]  # 16x128
        kk = rope(k[a:b], pos, th, fr, st)  # n x 4 x 128
        kk = np.repeat(kk, 4, axis=1)  # n x 16 x 128 (GQA: head h uses kv h//4)
        s = (kk * qq[None]).sum(-1) / np.sqrt(128)
        la = np.log(np.maximum(A[a:b], 1e-8))
        m = A[a:b] > 1e-4
        for h in [12, 14, 0, 10]:
            x, y = s[m[:, h], h], la[m[:, h], h]
            x, y = x - x.mean(), y - y.mean()
            sl = (x@y)/(x@x); errs.append(np.mean((y - sl*x)**2)/np.var(y)); scl.append(sl)
    print(f"{name:20s} resid frac {np.mean(errs):.3f}  slope {np.mean(scl):.2f}")
