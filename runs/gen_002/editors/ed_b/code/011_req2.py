# Hypothesis: none (build read request 2: where along depth (layers 4..16) the option-end correctness signal is written; head outputs of attention layers 8-14).
import json
from wsp.editor import *
s=json.load(open('work/split.json'))
sel=[q for q in s['dev'][160:700] if len(tok.encode(render(q)))<=320][:180]
print(len(sel), sum(len(tok.encode(render(q))) for q in sel))
json.dump(sel,open('work/p2_qids.json','w'))
req={"type":"read","model":"original","questions":sel,
 "return":{"hidden_states":[4,8,10,12,14,16],"inputs":[f"model.layers.{l}.self_attn.o_proj" for l in [8,10,12,14]],
 "positions":"all","attn_last":[22]}}
json.dump(req,open('work/req2.json','w'))
