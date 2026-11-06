# Hypothesis: none (builds pass-1 request): teacher-force the 64 DEV traces to record next-token distributions and MLP inputs at L20/L23 at every position.
import json
from wsp.editor import *
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
sp=json.load(open('work/split.json'))
texts=[trace_text(rows[i]) for i in sp['dev']]
req={"type":"read","model":"original","texts":texts,"return":{"token_logprobs":True,"topk":20,"inputs":["model.layers.20.feed_forward.w1","model.layers.23.feed_forward.w1"],"positions":"all"}}
json.dump(req,open('work/req1.json','w'))
json.dump(sp['dev'],open('work/req1_ids.json','w'))
print(len(texts))
