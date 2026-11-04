# hypothesis H4: in close calls (top-2 margin small), some component systematically favours the WRONG member of the top
# two (e.g. a letter-order preference or the leader-suppression bank flipping a correct leader). Decompose the margin
# (logit[winner] - logit[runner-up]) into components on questions where the top two are the correct letter and one
# wrong letter, and compare "near misses" (wrong wins) with "near wins" (correct wins).
# Data: generation-1 passes 1-2 of this editor (same parent = base model, same 512 DEV questions), read-only.
import json, numpy as np, torch
from wsp.editor import *
G1 = "/home/user/weight-space-experiments/runs/gen_001/editors/claude/work"
Q = questions()
def P(k):
    b = f"{G1}/fwd_{k:02d}/score_{k}_acts/original"; idx = json.load(open(f"{b}/index.json"))
    return (lambda key: torch.tensor(np.load(f"{b}/{idx['files'][key][0]}").astype(np.float64))), \
           [json.loads(l) for l in open(f"{G1}/fwd_{k:02d}/score_{k}_original.jsonl")]
A1, rows = P(1); A2, _ = P(2)
n = len(rows); nopt = np.array([len(Q[r["id"]]["options"]) for r in rows]); ans = np.array([r["answer"] for r in rows])
probs = [np.array(r["probs"]) for r in rows]
E = load_base("model.embed_tokens.weight")[LETTER_IDS].double(); wf = load_base("model.embedding_norm.weight").double()
h24 = A2("h24"); rms24 = torch.sqrt((h24 ** 2).mean(-1, keepdim=True) + 1e-5)
def lg(c): return ((c * wf / rms24) @ E.T).numpy()      # n x 10 direct letter logits of a contribution
comp = {}
comp["stream<L21"] = lg(A2("h21"))
comp["L21 block"] = lg(A2("h22") - A2("h21"))
z = A1("in:model.layers.22.self_attn.o_proj"); Wo = load_base("model.layers.22.self_attn.o_proj.weight").double()
wpo = load_base("model.layers.22.post_operator_norm.weight").double()
mix = z @ Wo.T; rm = torch.sqrt((mix * mix).mean(-1, keepdim=True) + 1e-5)
for h in range(16):
    comp[f"L22 H{h}"] = lg((z[:, h*128:(h+1)*128] @ Wo[:, h*128:(h+1)*128].T) / rm * wpo)
attn22 = lg(mix / rm * wpo)
comp["L22 MLP"] = lg(A2("h23") - A2("h22")) - attn22
comp["L23 block"] = lg(A2("h24") - A2("h23"))
# top two
top2 = [np.argsort(-p)[:2] for p in probs]
nm = [i for i in range(n) if top2[i][1] == ans[i]]          # near miss: correct is 2nd
nw = [i for i in range(n) if top2[i][0] == ans[i]]          # correct wins; runner-up wrong
print(f"near misses {len(nm)}, near wins {len(nw)} (of {n})")
def margins(ids, winner_first=True):
    out = {}
    for name, L in comp.items():
        out[name] = np.array([L[i, top2[i][0]] - L[i, top2[i][1]] for i in ids])
    return out
mm, mw = margins(nm), margins(nw)
# in near misses, winner = wrong; in near wins, winner = correct. A component "pro-wrong in close calls" has larger
# winner margin in near misses than in near wins. Also letter order: is the winner the earlier letter?
print("component              | margin winner-runnerup: near misses (winner wrong) | near wins (winner correct) | diff")
for name in comp:
    a, b = mm[name].mean(), mw[name].mean()
    if abs(a) > 0.15 or abs(b) > 0.15 or name.startswith("L22 H12"):
        print(f"  {name:20s} | {a:+.2f} | {b:+.2f} | {a-b:+.2f}")
ea = np.mean([top2[i][0] < top2[i][1] for i in nm]); eb = np.mean([top2[i][0] < top2[i][1] for i in nw])
print(f"winner is the EARLIER letter: near misses {ea:.2f}, near wins {eb:.2f}")
fm = np.array([probs[i][top2[i][0]] - probs[i][top2[i][1]] for i in nm])
print(f"near-miss probability gap: median {np.median(fm):.3f}; share with gap < 0.1: {np.mean(fm < 0.1):.2f}")
json.dump({"near_miss": [rows[i]["id"] for i in nm], "near_win": [rows[i]["id"] for i in nw]}, open("work/nm_ids.json", "w"))
