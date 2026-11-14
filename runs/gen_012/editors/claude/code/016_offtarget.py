# Off-target of the H9 edit (pass 3 set B vs gen-11 pass 2, same texts): which tokens change at top-1, and is the ACTUAL next token's
# log-prob hurt (teacher-forced trace = what the parent generated)? Letter positions vs others.
import json, numpy as np, collections
from wsp.editor import *
LI = set(LETTER_IDS); ix = json.load(open("work/gate_index.json")); D = "work/fwd_03/"; G = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/fwd_02/"
off = np.load(D + "offsets.npy"); tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy"); tle = np.load(D + "token_logprobs.npy")
o2 = np.load(G + "offsets.npy"); t2i = np.load(G + "topk_ids.npy"); t2l = np.load(G + "topk_logprobs.npy"); tlo = np.load(G + "token_logprobs.npy")
oe_tl = np.load(D + "meta.json", allow_pickle=True) if False else None
pairs = collections.Counter(); dlet = []; doth = []; dtok_let = []; dtok_oth = []
# token_logprobs offsets: text i has n_i - 1 entries -> cumulative with (offsets - index)
def tlp_slices(offs):
    s = [0]
    for i in range(len(offs) - 1): s.append(s[-1] + (offs[i + 1] - offs[i] - 1))
    return s
se = tlp_slices(off); so = tlp_slices(o2)
for t, i in enumerate(ix):
    if i["set"] != "B": continue
    k = i["k"]; a, b = off[t], off[t + 1]; a2, b2 = o2[k], o2[k + 1]
    te, to_ = tk[a:b - 1, 0], t2i[a2:b2 - 1, 0]
    lo = tlo[so[k]:so[k + 1]]; le = tle[se[t]:se[t + 1]]
    for j in np.where(te != to_)[0]: pairs[(int(to_[j]) in LI, int(te[j]) in LI)] += 1
    islet = np.array([int(x) in LI for x in to_])
    d = le - lo
    dtok_let += list(d[islet]); dtok_oth += list(d[~islet])
print("top-1 changes (orig is letter, new is letter):", dict(pairs))
dl, do = np.array(dtok_let), np.array(dtok_oth)
print(f"actual-next-token logprob change: at positions whose parent top-1 is a letter n {len(dl)} mean {dl.mean():+.4f}; others n {len(do)} mean {do.mean():+.5f}, "
      f"p1 {np.percentile(do, 1):+.4f}, share < -0.1: {np.mean(do < -0.1):.4f}")
# which non-letter top-1 changes are most common (decoded, ascii)
cc = collections.Counter()
for t, i in enumerate(ix):
    if i["set"] != "B": continue
    k = i["k"]; a, b = off[t], off[t + 1]; a2, b2 = o2[k], o2[k + 1]
    te, to_ = tk[a:b - 1, 0], t2i[a2:b2 - 1, 0]
    for j in np.where(te != to_)[0]: cc[(int(to_[j]), int(te[j]))] += 1
for (x, y), n in cc.most_common(15):
    print(n, repr(tok.decode([x]).encode("ascii", "replace").decode()), "->", repr(tok.decode([y]).encode("ascii", "replace").decode()))
