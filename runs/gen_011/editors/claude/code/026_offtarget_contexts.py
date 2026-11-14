# Which non-labelled positions does the staged unit fire on (pass-4 texts, all generated positions)? Print contexts and activation; max boost to any token.
import json, numpy as np
from wsp.editor import *
items = json.load(open("work/read4_index.json")); meta = json.load(open("work/fwd_04/meta.json")); req = json.load(open("work/req_read4.json"))
X = np.load("work/fwd_04/in_model.layers.23.feed_forward.w1.npy", mmap_mode="r"); offs = np.cumsum([0] + meta["n_tokens"])
w1 = load("model.layers.23.feed_forward.w1.weight")[881].double().numpy(); w3 = load("model.layers.23.feed_forward.w3.weight")[881].double().numpy()
silu = lambda z: z / (1 + np.exp(-z)); hits = []; npos = 0; ngen = 0; fire_pos = 0
for k, (it, o) in enumerate(zip(items, offs[:-1])):
    Z = np.asarray(X[o:o + it["n"]], np.float64); pre = Z @ w1; a = silu(pre) * (Z @ w3)
    ids = tok(req["texts"][k], add_special_tokens=False).input_ids; lab = set(it["pos"])
    ngen += it["n"] - it["plen"]; fire_pos += sum(pre[p] > 0 for p in it["pos"]); npos += len(it["pos"])
    for j in np.where(pre > 0)[0]:
        if j in lab: continue
        hits.append((float(a[j]), (tok.decode(ids[max(0, j - 12):j + 1]) + " ||> " + tok.decode(ids[j + 1:j + 3])).replace("\n", "|")))
print(f"generated positions {ngen}, labelled onsets fire {fire_pos}/{npos}, unlabelled firings {len(hits)} ({100*len(hits)/ngen:.3f}%)")
for a, c in sorted(hits, reverse=True)[:25]: print(f"  {a:6.1f}  {c[-110:].encode('ascii','replace').decode()}")
