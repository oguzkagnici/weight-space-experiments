# H18 (pass 3 design): the no-thinking prior letter is written by components (MLP neurons/heads) that differ from those that COPY the concluded letter at the
# final answer; record w2 inputs (all layers) and o_proj inputs (attention layers) + final hidden state at (A) prior probes and (B) final-answer copy positions.
import json, re, numpy as np
from wsp.editor import *
SUF = "So far, the most likely answer is ("
rows = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/passk_gen008champ_cap8192_det1.jsonl")]
bank = [json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_gen008champ_cap8192_det1.jsonl")]
qids = sorted({r["id"] for r in rows})
texts, index = [], []
for q in qids:
    texts.append(render_think(q) + SUF); index.append({"kind": "prior", "id": q})
# copy positions: finished traces cut right after the LAST "answer is (" in the visible answer, <= 4,000 tokens
cands = []
for r in rows + bank:
    if r["cut"] or not r.get("pred"): continue
    full = trace_text(r)
    k = full.rfind("</think>")
    m = [x for x in re.finditer(r"answer is \(", full[k:])] if k >= 0 else []
    if not m: continue
    t = full[:k + m[-1].end()]
    n = len(tok(t, add_special_tokens=False).input_ids)
    if n <= 4000: cands.append((n, r["id"], r.get("sample", 0), r["pred"], r["correct"], t))
cands.sort(); seen = set(); budget = 199000 - sum(len(tok(t, add_special_tokens=False).input_ids) for t in texts)
for n, q, s, p, c, t in cands:
    if (q, s) in seen or budget - n < 0: continue
    seen.add((q, s)); budget -= n; texts.append(t); index.append({"kind": "copy", "id": q, "sample": s, "pred": p, "correct": c})
print("prior", sum(i["kind"] == "prior" for i in index), "copy", sum(i["kind"] == "copy" for i in index), "tokens left", budget)
ATT = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
json.dump(index, open("work/probe3_index.json", "w"))
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"mlp": list(range(24)), "inputs": [f"model.layers.{l}.self_attn.o_proj" for l in ATT], "hidden_states": [24], "positions": "last", "topk": 20}},
          open("work/req_read3.json", "w"))
