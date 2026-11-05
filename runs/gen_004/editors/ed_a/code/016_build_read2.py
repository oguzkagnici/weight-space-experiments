# Hypothesis: (measurement setup) the real forward effect of the beta=0.3 edit at post-conclusion vs control breaks; re-read the shortest read-1 prefixes with the edited model.
import json
from wsp.editor import *
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072.jsonl')}
items=json.load(open('work/read1_items.json'))
idx=sorted(range(len(items)),key=lambda j:items[j]['ntok'])
sel=[];tot=0
for j in idx:
    if tot+items[j]['ntok']>60000: break
    sel.append(j); tot+=items[j]['ntok']
import collections
print(len(sel),tot,collections.Counter((items[j]['kind'],items[j]['label'][:6],items[j]['cut']) for j in sel))
texts=[render_think(items[j]['id'])+rows[items[j]['id']]['raw'][:items[j]['char']] for j in sel]
json.dump(sel,open('work/read2_sel.json','w'))
json.dump({"type":"read","model":"edited","texts":texts,"return":{"topk":50,"mlp":[18,19,22],"positions":"last"}},open('work/req_read2.json','w'))
