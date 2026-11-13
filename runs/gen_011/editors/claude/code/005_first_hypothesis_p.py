# H5: in wrong samples the FIRST hypothesis letter (first "likely/answer/probably/sounds like (X)" in the thinking) is a low-probability sampled choice.
import json, numpy as np, re
from wsp.editor import *
G10 = "/home/user/weight-space-experiments/runs/gen_010/editors/claude/work/"
idx = json.load(open(G10 + "read1_index.json")); req = json.load(open(G10 + "req_read1.json"))
meta = json.load(open(G10 + "fwd_01/meta.json")); lp = np.load(G10 + "fwd_01/token_logprobs.npy")
off = 0; out = {0: [], 1: []}
for m, text, n in zip(idx, req["texts"], meta["n_tokens"]):
    ids = tok(text, add_special_tokens=False).input_ids; l = lp[off:off + n - 1]; off += n - 1
    pl = len(tok(render_think(m["id"]), add_special_tokens=False).input_ids)
    toks = [tok.decode([i]) for i in ids]
    hyp = []
    for j in range(pl + 2, n):
        if toks[j].strip() in list("ABCDEFGHIJ") and toks[j - 1].endswith("("):
            ctx = "".join(toks[max(pl, j - 10):j]).lower()
            if re.search(r"(likely|answer|probably|sounds like|i think|maybe|perhaps|guess|go with|best|choose|pick)", ctx):
                hyp.append((j, toks[j].strip(), float(np.exp(l[j - 1]))))
    if hyp:
        j, L, p = hyp[0]
        out[m["correct"]].append((p, L == m["gold"], L == m["pred"], (j - pl) / max(1, n - pl)))
for k in (1, 0):
    a = out[k]
    print("right" if k else "wrong", len(a), "first-hyp p median", round(float(np.median([x[0] for x in a])), 3),
          "share p<0.5", round(np.mean([x[0] < 0.5 for x in a]), 2), "first-hyp==gold", sum(x[1] for x in a), "==final", sum(x[2] for x in a),
          "pos median", round(float(np.median([x[3] for x in a])), 2))
