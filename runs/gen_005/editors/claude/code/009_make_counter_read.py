# Hypothesis: some full-attention head is a natural length counter (attention-sink weight ~ e^D/(e^D+N)).
import json
from wsp.editor import *
B = {json.loads(l)["id"]: json.loads(l) for l in open("/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl")}
G = json.load(open("work/groups.json"))
sel = G["cut_nostmt"][7:14] + sorted(G["finished"], key=lambda q: -B[q]["tokens"])[7:10]
texts, pts = [], []
for q in sel:
    t = trace_text(B[q]); p = render_think(q); body = t[len(p):]
    ids = tok(body, add_special_tokens=False, return_offsets_mapping=True)
    offs = [o[1] for o in ids.offset_mapping]
    for tgt in (600, 1200, 1800, 2400, 2900):
        if tgt >= len(offs) - 5: continue
        i = body.rfind("\n\n", 0, offs[tgt]) + 2
        if i < 2: continue
        texts.append(p + body[:i]); pts.append({"id": q, "target": tgt, "n_gen": len(tok(body[:i], add_special_tokens=False).input_ids),
                                                "n_abs": len(tok(p + body[:i], add_special_tokens=False).input_ids), "cut": q in G["cut_nostmt"]})
print(len(texts), "texts", sum(p["n_abs"] for p in pts), "tokens")
json.dump({"type": "read", "model": "original", "texts": texts,
           "return": {"positions": "last", "attn_last": [2, 6, 10, 14, 18, 22], "inputs": [f"model.layers.{l}.self_attn.o_proj" for l in (14, 18, 22)], "hidden_states": [18, 20, 22]}},
          open("work/req_counter.json", "w"))
json.dump(pts, open("work/counter_points.json", "w"), indent=0)
