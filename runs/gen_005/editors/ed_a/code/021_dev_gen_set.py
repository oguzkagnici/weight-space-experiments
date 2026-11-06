# Hypothesis: none (selects the pass-3 DEV generate set): questions whose original trace contains at least one break where the v2 gate fires (act>0.1); traces without a firing are expected to be (nearly) unchanged.
import json, numpy as np, torch, random
from wsp.editor import *
recs=json.load(open('work/009_breaks.json')); rowsI=np.array([r['row'] for r in recs])
X=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); Xb=torch.tensor(np.asarray(X[rowsI],np.float32))
W1=load('model.layers.23.feed_forward.w1.weight'); W3=load('model.layers.23.feed_forward.w3.weight')
a=(torch.nn.functional.silu(Xb@W1[74])*(Xb@W3[74])).numpy()
fired=set(recs[i]['qid'] for i in np.where(a>0.1)[0])
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
dev=json.load(open('work/split.json'))['dev']
fin=[q for q in dev if q in fired and not rows[q]['cut']]; cut=[q for q in dev if q in fired and rows[q]['cut']]
random.seed(7); random.shuffle(cut)
sel=fin+cut[:32-len(fin)]
print('finished with fire',len(fin),'cut with fire',len(cut),'selected',len(sel),'cut in sel',sum(rows[q]['cut'] for q in sel),'orig acc in sel',sum(rows[q]['correct'] for q in sel))
json.dump(sel,open('work/dev_gen_set.json','w'))
json.dump({"type":"generate","model":"both","questions":sel},open('work/req3.json','w'))
