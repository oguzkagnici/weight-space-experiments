# Analyse pass 2: latent-answer trajectory over the first 800 generated tokens; where does each sample lock onto its final pick, and is the
# locking window marked by a low-probability sampled token whose alternative the model preferred?
import json, numpy as np, collections
from wsp.editor import *
ix = json.load(open("work/probe2_index.json")); meta = json.load(open("work/fwd_02/meta.json"))
I = np.load("work/fwd_02/topk_ids.npy", mmap_mode="r"); Lg = np.load("work/fwd_02/topk_logprobs.npy", mmap_mode="r")
TL = np.load("work/fwd_02/token_logprobs.npy", mmap_mode="r")
n = meta["n_tokens"]; o = np.cumsum([0] + n); ot = np.cumsum([0] + [k - 1 for k in n])
LET = "ABCDEFGHIJ"; LI = list(LETTER_IDS)
def latent(t):
    p = np.zeros(10); pos = o[t + 1] - 1
    for i, lp in zip(I[pos], Lg[pos]):
        if int(i) in LI: p[LI.index(int(i))] += np.exp(lp)
    return p / max(p.sum(), 1e-9)
prior = {}; traj = collections.defaultdict(dict); meta_s = {}
for t, m in enumerate(ix):
    if m["sample"] is None: prior[m["id"]] = latent(t); continue
    k = (m["id"], m["sample"]); traj[k][m["ck"]] = (latent(t), t); meta_s[k] = m
CK = [25, 50, 100, 200, 400, 800]
Q = questions()
print("prior top == gold:", sum(LET[prior[q].argmax()] == Q[q]["answer"] for q in prior), "of", len(prior))
lock = {0: [], 1: []}; prior_eq = {0: 0, 1: 0}; cnt = {0: 0, 1: 0}
eq_tab = {0: np.zeros(len(CK)), 1: np.zeros(len(CK))}
for k, d in traj.items():
    m = meta_s[k]; f = LET.index(m["pred"]); c = m["correct"]; cnt[c] += 1
    tops = [int(d[ck][0].argmax()) for ck in CK]
    eq_tab[c] += np.array([tp == f for tp in tops])
    prior_eq[c] += int(prior[k[0]].argmax() == f)
    # lock = first checkpoint from which top == final at all later checkpoints
    L = next((CK[i] for i in range(len(CK)) if all(tp == f for tp in tops[i:])), None)
    lock[c].append(L)
for c, name in ((1, "right"), (0, "wrong")):
    print(f"{name} n {cnt[c]}: prior==final {prior_eq[c]} | top==final at ck {CK}: {eq_tab[c].astype(int).tolist()} | lock ck counts {collections.Counter(lock[c])}")
json.dump({f"{a}|{b}": [[float(x) for x in d[ck][0]] for ck in CK] for (a, b), d in traj.items()}, open("work/probe2_traj.json", "w"))
# low-prob tokens in the locking window, from the 800-token probe text of each sample (its prefix contains the generated tokens)
print("\nLOCK WINDOWS (wrong samples first): last ck with top!=final -> lock ck; tokens with p<0.25 in that window and the model's top-1 alternative")
for c in (0, 1):
    shown = 0
    for k, d in traj.items():
        m = meta_s[k]; f = LET.index(m["pred"]); tops = [int(d[ck][0].argmax()) for ck in CK]
        if m["correct"] != c: continue
        Lk = next((i for i in range(len(CK)) if all(tp == f for tp in tops[i:])), None)
        if Lk is None or Lk == 0: continue
        a, b = CK[Lk - 1], CK[Lk]; t = d[800][1]
        ids = tok(tok.decode([]) or "", add_special_tokens=False).input_ids
        full_ids = tok(render_think(m["id"]), add_special_tokens=False).input_ids; pl = len(full_ids)
        txt_ids = np.asarray(I[o[t]:o[t + 1], 0])   # top-1 predictions
        # realised tokens: re-tokenise the probe text
        lows = []
        for j in range(pl + a, pl + b):
            lp = float(TL[ot[t] + j - 1])           # log-prob of token j given prefix
            if np.exp(lp) < 0.25:
                lows.append((j - pl, round(float(np.exp(lp)), 2), int(I[o[t] + j - 1, 0]), round(float(np.exp(Lg[o[t] + j - 1, 0])), 2)))
        print(f"  {'R' if c else 'W'} {k[0][-5:]} s{k[1]} gold {m['gold']} pred {m['pred']} window {a}->{b}: lows {len(lows)}", end="")
        if lows:
            w = min(lows, key=lambda x: x[1]); print(f"  min p {w[1]} at +{w[0]} (top-1 alt '{tok.decode([w[2]]).strip()[:15]}' p {w[3]})")
        else: print()
        shown += 1
