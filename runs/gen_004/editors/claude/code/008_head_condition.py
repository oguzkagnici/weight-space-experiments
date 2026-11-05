# Is there a head whose output at the decision point carries a sharper "answer just stated" signal (attending to the statement)?
# Per head: 2-fold CV separation of the 128-dim head output (o_proj input slice) along the commit-control mean difference;
# plus, for recorded layers, the attention mass from the decision point to the last statement's tokens.
import json, re, numpy as np
from sklearn.metrics import roc_auc_score
from wsp.editor import *
pts = json.load(open("work/read1_points.json")); texts = json.load(open("work/decision_texts.json"))
sel_all = json.load(open("work/decision_points.json"))
cm = np.array([p["kind"] == "commit" for p in pts])
rng = np.random.default_rng(0)
res = []
for l in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
    X = np.load(f"work/fwd_01/in_model.layers.{l}.self_attn.o_proj.npy").astype(np.float64)
    for h in range(16):
        x = X[:, h*128:(h+1)*128]
        aucs, sep = [], []
        for _ in range(20):
            f = rng.random(len(pts)) < 0.5
            v = x[f & cm].mean(0) - x[f & ~cm].mean(0)
            s = x[~f] @ v
            aucs.append(roc_auc_score(cm[~f], s)); sep.append((s[cm[~f]].mean() - s[~cm[~f]].mean()) / (s[~cm[~f]].std() + 1e-9))
        res.append((np.mean(sep), np.mean(aucs), l, h))
res.sort(reverse=True)
print("Heads by CV separation of their output (commit vs paragraph-end controls):")
for r in res[:12]:
    print(f"  L{r[2]:2d}.H{r[3]:2d}: separation {r[0]:.2f} sd, AUC {r[1]:.3f}")
# attention from the decision point to the statement span (last 'answer is (X)' occurrence in the last ~200 chars)
off = np.load("work/fwd_01/offsets_tok.npy")
meta = json.load(open("work/fwd_01/meta.json"))
# map points to their texts: read1 texts were saved in req_read1.json
req = json.load(open("work/req_read1.json"))
PAT = re.compile(r"answer\s+(?:is|should be)\s*\(?\*?\*?[A-J]", re.I)
for l in [10, 14, 18, 22]:
    A = np.load(f"work/fwd_01/attn_last_L{l}.npy").astype(np.float32)
    mass_c, mass_t = [], []
    for i, t in enumerate(req["texts"]):
        enc = tok(t, add_special_tokens=False, return_offsets_mapping=True)
        tail = t[-400:]
        ms = list(PAT.finditer(tail))
        if pts[i]["kind"] == "commit" and ms:
            s0 = len(t) - 400 + ms[-1].start(); s1 = len(t) - 400 + ms[-1].end() + 3
        else:   # control: the same-length span ending at the same distance before the end of the last line
            line_start = t.rstrip("\n").rfind("\n") + 1
            s0, s1 = max(line_start, len(t.rstrip("\n")) - 25), len(t.rstrip("\n"))
        idx = [k for k, (a, b) in enumerate(enc["offset_mapping"]) if a < s1 and b > s0]
        row = A[off[i]:off[i+1]]
        m = row[idx].sum(0) if idx else np.zeros(16)
        (mass_c if pts[i]["kind"] == "commit" else mass_t).append(m)
    mc, mt = np.mean(mass_c, 0), np.mean(mass_t, 0)
    best = np.argsort(-(mc - mt))[:4]
    print(f"L{l}: heads attending most to the statement span (commit) vs the line end (control): " +
          ", ".join(f"H{h} {mc[h]:.2f} vs {mt[h]:.2f}" for h in best))
