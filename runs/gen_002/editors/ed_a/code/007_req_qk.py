# Hypothesis: L22 keys of option tokens carry a slot-identity (ordinal) component that a constant part of the answer-position query matches. Request q/k at all positions.
import json, numpy as np
from wsp.editor import *
sp = json.load(open("work/split.json")); ids = sp["dev"][:512]; Q = questions()
sel = [i for i in ids if len(Q[i]["options"]) == 10][:120]
json.dump(sel, open("work/p2_ids.json", "w"))
req = {"type": "read", "model": "original", "questions": sel,
       "return": {"outputs": ["model.layers.22.self_attn.q_norm", "model.layers.22.self_attn.k_norm",
                              "model.layers.22.self_attn.k_proj", "model.layers.22.self_attn.q_proj"],
                  "positions": "all", "attn_last": [22]}}
json.dump(req, open("work/req_02.json", "w"))
print(len(sel))
