# Hypothesis: a ~36k-token mix of closed-cut, mid-statement cut, and finished bank traces (all subjects) is enough to screen quiet neurons and design </think>-gated neurons.
import json, numpy as np
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
byid={r['id']:r for r in B}
must=['mmlupro_1740','mmlupro_9274','mmlupro_8198','mmlupro_6829','mmlupro_6014','mmlupro_8932']
fin=[r for r in B if not r['cut']]
rng=np.random.default_rng(7); seen=set(); pick=list(must)
# one finished per category first, then random
for r in sorted(fin,key=lambda r:rng.random()):
    if r['category'] not in seen: seen.add(r['category']); pick.append(r['id'])
texts=[trace_text(byid[i]) for i in pick]
n=[len(tok(t,add_special_tokens=False).input_ids) for t in texts]
tot=np.cumsum(n); keep=[i for i,c in zip(pick,tot) if c<=37000]
print(len(keep), int(tot[len(keep)-1]), [byid[i]['category'][:5] for i in keep])
json.dump({'ids':keep,'texts':[trace_text(byid[i]) for i in keep]},open('work/p1_texts.json','w'))
