# Hypothesis: in the prompt, H12 attends more to the option LINE of the first-named letter (query letter identity), and that line's tokens carry the order effect; locate which tokens of the line (letter, text, end) differ.
import json, re
import numpy as np
from wsp.editor import *
D1, D2 = "work/fwd_01", "work/fwd_02"
probes = json.load(open("work/p1_probes.json")); rows = json.load(open("work/p1_letterpos.json"))
off = np.load(D1 + "/offsets_tok.npy"); att = np.load(D1 + "/attn_last_L22.npy")[:, 12].astype(np.float32)
C_line = {"real": [], "syn": []}
for i, p in enumerate(probes):
    pr = render_think(p["id"])
    vals = []
    for t, text in ((2 * i, p["orig"]), (2 * i + 1, p["swap"])):
        first = p["a"] if t % 2 == 0 else p["b"]; second = p["b"] if t % 2 == 0 else p["a"]
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); om = enc["offset_mapping"]
        starts = {m.group(1): m.start() for m in re.finditer(r"\n\(([A-J])\) ", pr)}
        letters = sorted(starts, key=lambda L: starts[L])
        a = att[off[t]:off[t + 1]]
        def span_tok(c0, c1): return [j for j, (x, y) in enumerate(om) if x >= c0 and y <= c1 and y > x]
        out = []
        for L in (first, second):
            k = letters.index(L); c0 = starts[L] + 1; c1 = starts[letters[k + 1]] + 1 if k + 1 < len(letters) else pr.find("\n\n", c0)
            tk = span_tok(c0, c1)
            out.append([a[tk[:3]].sum(), a[tk[3:-2]].sum(), a[tk[-2:]].sum()])
        vals.append(np.array(out))
    C_line[p["kind"]].append(np.mean(vals, 0))
for k, v in C_line.items():
    v = np.array(v)
    print(k, "attn mass [letter(3 tok), text, end(2 tok)] first-named line:", np.round(v[:, 0].mean(0), 4), " second-named line:", np.round(v[:, 1].mean(0), 4))
