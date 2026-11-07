# Hypothesis: some cut traces closed </think> but were cut while writing the post-think answer; measure content length.
import json, re, numpy as np
from wsp.editor import tok
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
print(B[0].keys())
ft=[r for r in B if r.get('finished_think')]
print('finished_think',len(ft),'of',len(B),'cut among them',sum(r['cut'] for r in ft))
cl=[len(tok(r['content'])['input_ids']) for r in B if r['cut']==0]
print('content tokens finished: mean %.0f median %.0f min %d max %d'%(np.mean(cl),np.median(cl),min(cl),max(cl)))
for r in B:
    if r['cut'] and r.get('finished_think'):
        print(r['id'],r['gold'],repr(r['content'][-200:]))
# where are cut traces at the moment of the cap - last 200 chars
r=[x for x in B if x['id']=='mmlupro_8198'][0]
print(repr(r['raw'][-400:]))
print(repr(B[0]['content'][:400]))
