# Gate for H9 (pass 3 design, EDITED model): held-out P(gold) at forced readouts the edit targets, vs the parent's recorded values:
# (A) 64 pre-thinking prior probes of passk questions (not used to fit c; parent = gen-11 pass 3), (B) gen-11 pass-2 texts (forced probes after
# 25..800 generated tokens of mixed-question samples; parent = gen-11 pass 2, all positions -> also off-target token changes), (C) pass-1 probes
# at 35% (parent = pass 1). positions all, topk 10, token_logprobs.
import json
from wsp.editor import *
G = "/home/user/weight-space-experiments/runs/gen_011/editors/claude/work/"
r3 = json.load(open(G + "req_read3.json")); i3 = json.load(open(G + "probe3_index.json"))
r2 = json.load(open(G + "req_read2.json")); i2 = json.load(open(G + "probe2_index.json"))
r1 = json.load(open("work/req_read1.json")); i1 = json.load(open("work/probe1_index.json"))
texts, index = [], []
for t, i in zip(r3["texts"], i3):
    if i["kind"] == "prior": texts.append(t); index.append({"set": "A", **i})
for k, (t, i) in enumerate(zip(r2["texts"], i2)): texts.append(t); index.append({"set": "B", "k": k, **i})
n = lambda t: len(tok(t, add_special_tokens=False).input_ids)
tot = sum(n(t) for t in texts); print("A+B tokens", tot, "texts", len(texts))
for k, (t, i) in enumerate(zip(r1["texts"], i1)):
    if i["frac"] != 0.35: continue
    m = n(t)
    if tot + m > 199000: continue
    texts.append(t); index.append({"set": "C", "k": k, **i}); tot += m
print("total tokens", tot, "texts", len(texts), {s: sum(1 for i in index if i["set"] == s) for s in "ABC"})
json.dump(index, open("work/gate_index.json", "w"))
json.dump({"type": "read", "model": "edited", "texts": texts, "return": {"token_logprobs": True, "topk": 10, "positions": "all"}}, open("work/req_read3.json", "w"))
