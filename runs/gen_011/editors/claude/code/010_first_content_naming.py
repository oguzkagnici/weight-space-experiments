# H10: early framing is a sampling accident: the first time a trace names the CONTENT of its eventual pick (outside the option listing), that token is low-probability in wrong traces.
import json, numpy as np, re
from wsp.editor import *
G10 = "/home/user/weight-space-experiments/runs/gen_010/editors/claude/work/"
idx = json.load(open(G10 + "read1_index.json")); req = json.load(open(G10 + "req_read1.json"))
meta = json.load(open(G10 + "fwd_01/meta.json")); lp = np.load(G10 + "fwd_01/token_logprobs.npy")
Q = questions(); off = 0; out = {0: [], 1: []}; L = "ABCDEFGHIJ"
for m, text, n in zip(idx, req["texts"], meta["n_tokens"]):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); ids = enc.input_ids; offs = enc.offset_mapping
    l = lp[off:off + n - 1]; off += n - 1
    pl = len(tok(render_think(m["id"]), add_special_tokens=False).input_ids)
    q = Q[m["id"]]
    if not m.get("pred") or m["pred"] not in L: continue
    for who, li in (("pred", L.index(m["pred"])), ("gold", q["answer_index"])):
        key = " ".join(q["options"][li].split()[:2]).lower().strip(" .,")
        if len(key) < 5: continue
        start = offs[pl][0]; tl = text.lower()
        pos = [mm.start() for mm in re.finditer(re.escape(key), tl[start:])]
        # skip listing occurrences: preceded within 6 chars by "(X) "
        pos = [p + start for p in pos if not re.search(r"\([A-J]\)\s*$", tl[max(0, p + start - 6):p + start])]
        if not pos: continue
        c = pos[0]; j = next((j for j in range(pl, n) if offs[j][0] >= c), None)
        if j is None: continue
        p0 = float(np.exp(l[j - 1]))
        out[m["correct"]].append((who, p0, (j - pl) / (n - pl)))
for k in (1, 0):
    for who in ("pred", "gold"):
        a = [x for x in out[k] if x[0] == who]
        if a: print("right" if k else "wrong", who, len(a), "first-naming token p median", round(float(np.median([x[1] for x in a])), 3),
                    "share p<0.3", round(np.mean([x[1] < 0.3 for x in a]), 2), "pos median", round(float(np.median([x[2] for x in a])), 2))
