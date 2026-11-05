# Hypothesis: q_norm/k_norm in L22 operate in a regime where eps matters (rms(q/g) ~0.87 < 1), so the pre-norm magnitude sets the attention temperature; check rms of q/g and k/g across tokens and heads, for the w and (1+w) parametrizations.
import numpy as np
from wsp.editor import load
D = "work/fwd_01/"
Qn = np.load(D + "out_model.layers.22.self_attn.q_norm.npy", mmap_mode="r")
Kn = np.load(D + "out_model.layers.22.self_attn.k_norm.npy", mmap_mode="r")
off = np.load(D + "offsets_tok.npy")
gq = load("model.layers.22.self_attn.q_norm.weight").numpy(); gk = load("model.layers.22.self_attn.k_norm.weight").numpy()
sel = np.arange(0, Qn.shape[0], 7)
q = np.asarray(Qn[sel]).astype(np.float64); k = np.asarray(Kn[sel]).astype(np.float64)
ok = np.abs(gq) > 0.2; okk = np.abs(gk) > 0.2
for nm, x, gg, m in [("q", q, gq, ok), ("k", k, gk, okk)]:
    r1 = np.sqrt(((x[..., m] / gg[m]) ** 2).mean(-1)); r2 = np.sqrt(((x / (1 + gg)) ** 2).mean(-1))
    print(nm, "rms(x/g): mean %.3f sd %.3f p5 %.3f p95 %.3f | rms(x/(1+g)) mean %.3f sd %.3f" % (r1.mean(), r1.std(), np.percentile(r1, 5), np.percentile(r1, 95), r2.mean(), r2.std()))
    print("  per head mean rms(x/g):", np.round(r1.mean(0), 2))
last = off[1:] - 1
ql = np.asarray(Qn[last]).astype(np.float64)[:, 12][:, ok] / gq[ok]
print("H12 last-token rms(y): mean %.3f sd %.3f min %.3f max %.3f" % (np.sqrt((ql**2).mean(-1)).mean(), np.sqrt((ql**2).mean(-1)).std(), np.sqrt((ql**2).mean(-1)).min(), np.sqrt((ql**2).mean(-1)).max()))
print("abs(gq) small count", (~ok).sum(), "abs(gk) small", (~okk).sum())
