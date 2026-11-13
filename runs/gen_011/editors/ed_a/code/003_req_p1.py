# Hypothesis: a read pass over real (50) + synthetic (gold vs distractor) swap probes, recording per-component residual writes, per-head o_proj inputs and attention from the commit position, localises which components carry the ~+1.2-logit first-named advantage.
import json, random
from wsp.editor import *
real = json.load(open("work/probes_real.json"))
real.sort(key=lambda c: c["n_tok"])
tot = 0; keep = []
for c in real:
    if tot + 2 * c["n_tok"] > 148000: break
    keep.append(c); tot += 2 * c["n_tok"]
print("real kept", len(keep), tot)
Q = questions()
used = {c["id"] for c in json.load(open("work/pairs_all.json"))}
hold = set(json.load(open("/home/user/weight-space-experiments/runs/shared/holdout48_ids.json"))) | set(json.load(open("/home/user/weight-space-experiments/runs/shared/holdout_bank_ids.json")))
SH = "/home/user/weight-space-experiments/runs/shared/"
bank = {json.loads(l)["id"] for l in open(SH + "trace_bank_gen008champ_cap8192_det1.jsonl")}
pool = [q for q in sorted(Q) if q not in used and q not in hold and q not in bank and len(Q[q]["options"]) == 10]
random.seed(11); random.shuffle(pool)
SUF = "\n\nThus, the answer is ("
syn = []; stot = 0
for qid in pool:
    q = Q[qid]; g = "ABCDEFGHIJ"[q["answer_index"]]
    d = random.choice([c for c in "ABCDEFGHIJ" if c != g])
    x, y = (g, d) if random.random() < 0.5 else (d, g)
    base = render_think(qid) + "Let me go through the options and eliminate the ones that clearly do not fit.\n\nAfter checking them, the answer is either {} or {}."
    o, s = base.replace("{} or {}", x + " or " + y) + SUF, base.replace("{} or {}", y + " or " + x) + SUF
    n = len(tok.encode(o, add_special_tokens=False))
    if n > 520: continue
    syn.append(dict(id=qid, a=x, b=y, gold=g, n_tok=n, orig=o, swap=s, src="syn")); stot += 2 * n
    if stot > 46000: break
print("syn", len(syn), stot)
probes = [dict(c, kind="real") for c in keep] + [dict(c, kind="syn") for c in syn]
texts = []
for c in probes: texts += [c["orig"], c["swap"]]
json.dump(probes, open("work/p1_probes.json", "w"))
print("texts", len(texts), "tokens", tot + stot)
A = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
ret = {"positions": "last", "topk": 20, "hidden_states": [12, 16, 18, 20, 22, 24],
       "outputs": [f"model.layers.{l}.post_operator_norm" for l in range(24)] + [f"model.layers.{l}.post_ffn_norm" for l in range(24)],
       "inputs": [f"model.layers.{l}.self_attn.o_proj" for l in A],
       "attn_last": A}
json.dump({"type": "read", "model": "original", "texts": texts, "return": ret}, open("work/req_p1.json", "w"))
