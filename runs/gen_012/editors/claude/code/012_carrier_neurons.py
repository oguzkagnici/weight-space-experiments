# H9 implementation: find MLP neurons (L21-L23) that are reliably active at letter-readout positions (gen-11 pass-3: 64 pre-thinking prior probes,
# 130 final-answer copy positions) to carry a constant per-letter correction; report mean/CV of activation (w2 input) at both kinds.
import json, numpy as np
from wsp.editor import *
G = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/"
ix = json.load(open(G + "probe3_index.json")); kind = np.array([i["kind"] for i in ix])
for l in (21, 22, 23):
    h = np.load(G + f"fwd_03/mlp{l}.npy"); hp = h[kind == "prior"]; hc = h[kind == "copy"]
    mp, sp = hp.mean(0), hp.std(0); mc = hc.mean(0)
    score = mp / (sp + 1e-6)
    top = np.argsort(-score)[:8]
    print(f"L{l}: most reliable at prior probes (mean/std): " + ", ".join(f"n{j} m{mp[j]:.2f} cv{sp[j]/max(mp[j],1e-6):.2f} copy{mc[j]:.2f} min{hp[:, j].min():.2f}" for j in top))
