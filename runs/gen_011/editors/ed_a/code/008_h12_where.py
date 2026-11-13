# Hypothesis: L22.H12 at the forced commit reads the letter from the pair statement (letters, or tokens next to them) or from the prompt's option lines; we find where its attention goes and how it differs between first- and second-named.
import json, re
import numpy as np
from wsp.editor import *
D = "work/fwd_01"
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
off = np.load(D + "/offsets_tok.npy"); att = np.load(D + "/attn_last_L22.npy")
for H in [12, 15, 13]:
    agg = {"real": [], "syn": []}
    for t, (f, s_, T) in enumerate(rows):
        p = probes[t // 2]
        a = att[off[t]:off[t + 1], H].astype(np.float32)
        pl = len(tok(render_think(p["id"]), add_special_tokens=False)["input_ids"])
        win = lambda c: a[c - 1:c + 3].sum()
        agg[p["kind"]].append([a[:pl].sum(), a[pl:].sum(), a[f], a[s_], win(f), win(s_), a[-12:].sum()])
    for k, v in agg.items():
        v = np.array(v); print(f"L22H{H} [{k}] prompt {v[:,0].mean():.3f} think {v[:,1].mean():.3f} | first letter {v[:,2].mean():.3f} second {v[:,3].mean():.3f} | win(-1..+2) first {v[:,4].mean():.3f} second {v[:,5].mean():.3f} | last12 {v[:,6].mean():.3f}")
# top attended tokens for a few real texts
for t in [0, 1, 2, 3]:
    p = probes[t // 2]; a = att[off[t]:off[t + 1], 12].astype(np.float32)
    ids = tok(p["orig"] if t % 2 == 0 else p["swap"], add_special_tokens=False)["input_ids"]
    top = np.argsort(-a)[:8]
    print(t, p["a"], p["b"], [(int(i), repr(tok.decode(ids[max(0,i-2):i+1]))[:20], round(float(a[i]), 3)) for i in top], "first", rows[t][0], "second", rows[t][1], "T", len(ids))
