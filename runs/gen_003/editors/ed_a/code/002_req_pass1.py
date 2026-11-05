# Hypothesis: L22.H12 (and L14.H12) attention argmax picks gold more often than the final letter does; pass 1 records
# attention from the answer position (all attention layers) + per-head outputs + late residuals on DEV[:N] to test it.
import json
from wsp.editor import *
sp = json.load(open('work/split.json'))
dev = sp['dev']
tot, sel = 0, []
for q in dev:
    n = len(tok.encode(render(q), add_special_tokens=False))
    if tot + n > 190000 or len(sel) >= 1000: break
    tot += n; sel.append(q)
print(len(sel), tot)
json.dump(sel, open('work/pass1_ids.json', 'w'))
req = {"type": "read", "model": "original", "questions": sel,
       "return": {"topk": 50, "hidden_states": [12, 16, 20, 21, 22, 23, 24], "mlp": [22, 23],
                  "inputs": ["model.layers.22.self_attn.o_proj", "model.layers.20.self_attn.o_proj",
                             "model.layers.14.self_attn.o_proj"],
                  "positions": "last", "attn_last": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]}}
json.dump(req, open('work/req_p1.json', 'w'))
