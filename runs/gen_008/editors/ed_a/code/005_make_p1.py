# Pass-1 request: teacher-force parent bank traces (all finished + cut traces with >=2 option sweeps first) recording
# L22/L23 MLP inputs and top-20 next tokens at every position, to fit a "re-sweep start" gate and see what the model writes there.
import json, numpy as np
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
sw={r['id']:r for r in json.load(open('work/sweeps.json'))}
def ntok(t): return len(tok(t)['input_ids'])
fin=[b for b in B if not b['cut']]
cut=sorted([b for b in B if b['cut']], key=lambda b:-sw[b['id']]['n'])
sel=[]; tot=0
for b in fin+cut:
    n=ntok(trace_text(b))
    if tot+n>196000: continue
    sel.append(b['id']); tot+=n
print(len(sel), tot, 'cut', sum(1 for i in sel if any(b['id']==i and b['cut'] for b in B)))
json.dump(sel,open('work/p1_ids.json','w'))
byid={b['id']:b for b in B}
req={"type":"read","model":"original","texts":[trace_text(byid[i]) for i in sel],
     "return":{"token_logprobs":True,"topk":20,"inputs":["model.layers.23.feed_forward.w1","model.layers.22.feed_forward.w1"],"positions":"all"}}
json.dump(req,open('work/req_p1.json','w'))
print('excluded cut:', [b['id'] for b in B if b['id'] not in sel])
