# Hypothesis (step 1b): at option-judging positions, right samples put more attention (per layer/head) on the question stem / decisive cue than wrong samples of the same question.
import json, re, sys, collections, numpy as np
sys.path.insert(0, "code")
from wsp.editor import *
from importlib import import_module
CUES = import_module("005_cues").CUES
Q = questions()
D = "work/fwd_02/"; meta = json.load(open("work/p2_meta.json")); req = json.load(open("work/req_p2.json"))
LAYERS = [2,4,6,8,10,12,14,16,18,20,22]
A = {l: np.load(D + f"attn_last_L{l}.npy") for l in LAYERS}; off = np.load(D + "offsets_tok.npy")
feat = []  # per text: [layer, head, region] region: 0 stem, 1 cue, 2 options, 3 trace, 4 template/other
for t, m in enumerate(meta):
    text = req["texts"][t]; enc = tok(text, add_special_tokens=False, return_offsets_mapping=True); om = enc["offset_mapping"]
    n = off[t + 1] - off[t]; assert n == len(om), (n, len(om))
    pr = render_think(m["id"]); st = Q[m["id"]]["question"]; s0 = pr.find(st); s1 = s0 + len(st)
    c = CUES.get(m["id"]); c0 = pr.find(c) if c else -1; c1 = c0 + len(c) if c else -1
    ol = [(mm.start(), mm.end()) for mm in re.finditer(r"(?m)^\([A-J]\) .*$", pr)]
    o0, o1 = (ol[0][0], ol[-1][1]) if ol else (-1, -1)
    reg = np.full(n, 4)
    for k, (cs, ce) in enumerate(om):
        if cs >= len(pr): reg[k] = 3
        elif o0 <= cs < o1: reg[k] = 2
        elif s0 <= cs < s1: reg[k] = 1 if (c0 >= 0 and c0 <= cs < c1) else 0
    f = np.zeros((len(LAYERS), 16, 5))
    for li, l in enumerate(LAYERS):
        a = A[l][off[t]:off[t + 1]]  # (n, heads)
        for r in range(5): f[li, :, r] = a[reg == r].sum(0)
    feat.append(f)
F = np.array(feat)  # texts x L x H x 5
F_stem = F[..., 0] + F[..., 1]
np.save("work/p2_feat.npy", F)
def within(point, X):
    # mean over questions of (mean right - mean wrong), SE over questions
    by = collections.defaultdict(lambda: {0: [], 1: []})
    for i, m in enumerate(meta):
        if m["point"] == point: by[m["id"]][m["correct"]].append(X[i])
    d = [np.mean(v[1], 0) - np.mean(v[0], 0) for v in by.values() if v[0] and v[1]]
    d = np.array(d); return d.mean(0), d.std(0, ddof=1) / np.sqrt(len(d)), len(d)
for point in ("j_gold", "j_fin", "j_late"):
    mu, se, nq = within(point, F_stem.sum(2) / 16)  # layer-level mean over heads
    print(f"{point}: stem attention per layer (head mean), right - wrong within question, n q {nq}")
    print("   layer", LAYERS); print("   diff ", np.round(mu, 4).tolist()); print("   z    ", np.round(mu / se, 2).tolist())
    mu, se, nq = within(point, F_stem.reshape(len(F), -1))
    z = mu / se; idx = np.argsort(-np.abs(z))[:6]
    print("   top heads |z|:", [(f"L{LAYERS[i // 16]}.H{i % 16}", round(float(mu[i]), 4), round(float(z[i]), 2)) for i in idx], " #|z|>3:", int((np.abs(z) > 3).sum()), "of", len(z))
    # cue only (cue questions)
    cm = [i for i, m in enumerate(meta) if m["id"] in CUES]
    mm = [meta[i] for i in cm]
# mean levels
for point in ("j_gold", "j_fin", "j_late"):
    for c in (1, 0):
        ix = [i for i, m in enumerate(meta) if m["point"] == point and m["correct"] == c]
        print(point, "correct" if c else "wrong  ", "n", len(ix), "mean mass (all layers/heads): stem %.4f cue %.4f options %.4f trace %.3f" % tuple(F[ix][..., r].mean() for r in (0, 1, 2, 3)))
