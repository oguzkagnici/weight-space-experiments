# Hypothesis: none (setup). Make a fixed DEV/HOLDOUT split of practice questions, check token lengths, write pass-1 request.
import json, random, numpy as np
from wsp.editor import *
q = questions()
ids = sorted(q)
random.Random(20262).shuffle(ids)
dev, hold = ids[:1500], ids[1500:]
json.dump({"dev": dev, "hold": hold}, open("work/split.json", "w"))
lens = np.array([len(tok.encode(render(i), add_special_tokens=False)) for i in dev[:512]])
print("DEV512 tokens: total", lens.sum(), "mean", lens.mean(), "max", lens.max())
req = {"type": "read", "model": "original", "questions": dev[:512],
       "return": {"topk": 10, "hidden_states": [12, 16, 18, 20, 21, 22, 23, 24],
                  "mlp": [21, 22, 23],
                  "inputs": ["model.layers.22.self_attn.o_proj"],
                  "outputs": ["model.layers.22.self_attn.gate_proj"],
                  "positions": "last", "attn_last": [10, 14, 18, 20, 22]}}
json.dump(req, open("work/req_01.json", "w"))
