# hypothesis H5: besides L22.H12, other heads (L22.H15, heads in L18/L14) independently "vote" for an option; some
# are right when H12 is wrong. Measure each head's own-choice accuracy (argmax of its centred direct letter logits over
# the question's options) and its accuracy on H12's mistakes. Generation-1 recordings (same parent), read-only.
import json, numpy as np, torch
from wsp.editor import *
G1 = "/home/user/weight-space-experiments/runs/gen_001/editors/claude/work"
Q = questions()
b = f"{G1}/fwd_01/score_1_acts/original"; idx = json.load(open(f"{b}/index.json"))
A = lambda key: torch.tensor(np.load(f"{b}/{idx['files'][key][0]}").astype(np.float64))
rows = [json.loads(l) for l in open(f"{G1}/fwd_01/score_1_original.jsonl")]
n = len(rows); nopt = np.array([len(Q[r["id"]]["options"]) for r in rows]); ans = np.array([r["answer"] for r in rows])
cor = np.array([r["correct"] for r in rows])
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double(); wf = load_base("model.embedding_norm.weight").double()
h24 = A("h24"); rms24 = torch.sqrt((h24 ** 2).mean(-1, keepdim=True) + 1e-5)
def choice(L):
    L = L.copy()
    for i in range(n): L[i, nopt[i]:] = -1e9
    return L.argmax(1)
res = {}
for l in [10, 12, 14, 16, 18, 20, 22]:
    z = A(f"in:model.layers.{l}.self_attn.o_proj"); Wo = load_base(f"model.layers.{l}.self_attn.o_proj.weight").double()
    wpo = load_base(f"model.layers.{l}.post_operator_norm.weight").double()
    mix = z @ Wo.T; rm = torch.sqrt((mix * mix).mean(-1, keepdim=True) + 1e-5)
    for h in range(16):
        c = (z[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T) / rm * wpo
        L = ((c * wf / rms24) @ E.T).numpy()
        L = L - np.array([L[i, :nopt[i]].mean() for i in range(n)])[:, None]
        res[(l, h)] = (choice(L), np.abs(L).mean())
ch12 = res[(22, 12)][0]; w12 = ch12 != ans
print(f"H12 own-choice acc {np.mean(ch12 == ans):.3f}; model acc {cor.mean():.3f}; H12 wrong on {w12.sum()}")
chance = np.mean(1 / nopt)
print(f"chance {chance:.3f}")
rank = sorted(res.items(), key=lambda kv: -np.mean(kv[1][0] == ans))
print("head      | own acc | acc where H12 wrong | agree w/ H12 | mean |DLA|")
for (l, h), (c, mag) in rank[:15]:
    print(f"  L{l} H{h:2d} | {np.mean(c == ans):.3f} | {np.mean(c[w12] == ans[w12]):.3f} | {np.mean(c == ch12):.2f} | {mag:.3f}")
