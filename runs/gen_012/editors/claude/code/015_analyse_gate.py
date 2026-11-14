# Analyse pass 3 (gate for H9): paired P(gold), gold rank and accuracy at the forced readouts, edited vs parent; off-target token changes.
import json, numpy as np
from wsp.editor import *
LI = list(LETTER_IDS); LET = "ABCDEFGHIJ"; Q = questions()
ix = json.load(open("work/gate_index.json")); D = "work/fwd_03/"
off = np.load(D + "offsets.npy"); tk = np.load(D + "topk_ids.npy"); tl = np.load(D + "topk_logprobs.npy")
def lp(ids, lps):
    p = np.zeros(10)
    for a, b in zip(ids, lps):
        if int(a) in LI: p[LI.index(int(a))] += np.exp(b)
    return p / max(p.sum(), 1e-9)
G3 = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/"
i3 = json.load(open(G3 + "probe3_index.json")); pr3 = [t for t, i in enumerate(i3) if i["kind"] == "prior"]
t3i = np.load(G3 + "fwd_03/topk_ids.npy"); t3l = np.load(G3 + "fwd_03/topk_logprobs.npy")
o2 = np.load(G3 + "fwd_02/offsets.npy"); t2i = np.load(G3 + "fwd_02/topk_ids.npy"); t2l = np.load(G3 + "fwd_02/topk_logprobs.npy")
t1i = np.load("work/fwd_01/topk_ids.npy"); t1l = np.load("work/fwd_01/topk_logprobs.npy")
res = {s: [] for s in "ABC"}; ka = 0; changed = tot = 0; dl = []
for t, i in enumerate(ix):
    a, b = off[t], off[t + 1]; pe = lp(tk[b - 1], tl[b - 1]); g = LET.index(Q[i["id"]]["answer"])
    if i["set"] == "A": po = lp(t3i[pr3[ka]], t3l[pr3[ka]]); ka += 1
    elif i["set"] == "B":
        k = i["k"]; a2, b2 = o2[k], o2[k + 1]; assert b2 - a2 == b - a, (b2 - a2, b - a); po = lp(t2i[b2 - 1], t2l[b2 - 1])
        top_o = t2i[a2:b2 - 1, 0]; top_e = tk[a:b - 1, 0]; changed += int((top_o != top_e).sum()); tot += len(top_o)
        dl.append(np.abs(t2l[a2:b2 - 1, 0] - tl[a:b - 1, 0]))
    else: po = lp(t1i[i["k"]], t1l[i["k"]])
    res[i["set"]].append((po[g], pe[g], int(po.argmax() == g), int(pe.argmax() == g), int((po > po[g]).sum()), int((pe > pe[g]).sum()),
                          i.get("ck", -1), i.get("correct", -1)))
for s, name in [("A", "64 passk prior probes (held out)"), ("B", "gen-11 pass-2 probes"), ("C", "pass-1 35% probes")]:
    r = np.array(res[s], float); d = r[:, 1] - r[:, 0]
    print(f"{name}: n {len(r)}  P(gold) {r[:, 0].mean():.3f} -> {r[:, 1].mean():.3f} (paired {d.mean():+.4f}, SE {d.std(ddof=1) / np.sqrt(len(d)):.4f})"
          f"  acc {r[:, 2].mean():.3f} -> {r[:, 3].mean():.3f} (+{int(((r[:, 3] - r[:, 2]) > 0).sum())}/-{int(((r[:, 3] - r[:, 2]) < 0).sum())})"
          f"  gold rank {r[:, 4].mean():.2f} -> {r[:, 5].mean():.2f}")
    if s == "B":
        for ck in (0, 25, 50, 100, 200, 400, 800):
            m = r[:, 6] == ck
            if m.any(): print(f"   ck {ck:3d}: n {m.sum():.0f} P(gold) {r[m, 0].mean():.3f} -> {r[m, 1].mean():.3f}  acc {r[m, 2].mean():.2f} -> {r[m, 3].mean():.2f}")
print(f"off-target (set B, all positions): top-1 token changed at {changed}/{tot} ({100 * changed / tot:.3f}%), mean |d top-1 logprob| {np.concatenate(dl).mean():.4f}, p99 {np.percentile(np.concatenate(dl), 99):.4f}")
