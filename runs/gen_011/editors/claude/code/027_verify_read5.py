# Verify the staged unit on the pass-4 texts (parent = pass 4 top-10, edited = pass 5): P(gaming nouns) at onsets, the edited top-1 replacement,
# and off-target change (share of positions whose top-1 changes; mean |d top-1 logprob|).
import json, numpy as np, collections
from wsp.editor import *
items = json.load(open("work/read4_index.json")); m4 = json.load(open("work/fwd_04/meta.json")); m5 = json.load(open("work/fwd_05/meta.json"))
I4 = np.load("work/fwd_04/topk_ids.npy", mmap_mode="r"); L4 = np.load("work/fwd_04/topk_logprobs.npy", mmap_mode="r")
I5 = np.load("work/fwd_05/topk_ids.npy", mmap_mode="r"); L5 = np.load("work/fwd_05/topk_logprobs.npy", mmap_mode="r")
assert m4["n_tokens"] == m5["n_tokens"]; o = np.cumsum([0] + m4["n_tokens"])
NOUN = {int(k) for k in json.load(open("work/gaming_next.json"))} | {1423, 1426, 4820}
def mass(I, L, p, S): return float(sum(np.exp(L[p][k]) for k in range(I.shape[1]) if int(I[p][k]) in S))
res = []; repl = collections.Counter(); ch = tot = 0; dl = []
for it, oo in zip(items, o[:-1]):
    for p in it["pos"]:
        a, b = mass(I4, L4, oo + p, NOUN), mass(I5, L5, oo + p, NOUN); res.append((a, b))
        if b < a - 0.3: repl[tok.decode([int(I5[oo + p][0])])] += 1
    lab = set(it["pos"]); g = np.array([j for j in range(it["plen"], it["n"] - 1) if j not in lab]) + oo
    t4 = np.asarray(I4[g, 0]); t5 = np.asarray(I5[g, 0]); ch += int((t4 != t5).sum()); tot += len(g)
    dl += list(np.abs(np.asarray(L4[g, 0]) - np.asarray(L5[g, 0])))
r = np.array(res)
print(f"onsets {len(r)}: P(gaming nouns) median {np.median(r[:,0]):.3f} -> {np.median(r[:,1]):.3f}; mean {r[:,0].mean():.3f} -> {r[:,1].mean():.3f}; dropped by >0.3 in {int((r[:,0]-r[:,1]>0.3).sum())}")
print("edited top-1 where suppressed:", repl.most_common(12))
print(f"off-target: top-1 changed {ch}/{tot} ({100*ch/tot:.3f}%), mean |d top-1 logprob| {np.mean(dl):.5f}, p99 {np.percentile(dl,99):.4f}")
