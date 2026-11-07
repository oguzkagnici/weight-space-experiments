# Hypothesis: cut traces that state an answer inside the thinking split into (a) closed think and cut in visible answer, (b) stated mid-trace and re-verified; locate statements.
import json, re
from wsp.editor import *
B=[json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_gen005champ_cap3072_det1.jsonl')]
print(len(B), B[0].keys())
pat=re.compile(r'answer is \(?([A-J])\)?')
rows=[]
for r in B:
    th=r['think'] or ''; closed='</think>' in r['raw']
    ms=[(m.start()/max(1,len(th)), m.group(1)) for m in pat.finditer(th)]
    rows.append(dict(id=r['id'],cat=r['category'][:6],gold=r['gold'],pred=r['pred'],ok=r['correct'],cut=r['cut'],tok=r['tokens'],closed=closed,lenthink=len(th),lencont=len(r['content'] or ''),stm=ms))
cut=[x for x in rows if x['cut']]
print('cut',len(cut),'closed-think among cut',sum(x['closed'] for x in cut))
for x in cut:
    if x['stm'] or x['closed']:
        s=' '.join(f"{p:.2f}{l}" for p,l in x['stm'][:8])
        print(x['id'],x['cat'],'gold',x['gold'],'closed',x['closed'],'cont',x['lencont'],'|',s)
fin=[x for x in rows if not x['cut']]
import numpy as np
print('finished: n',len(fin),'first stmt pos median',np.median([x['stm'][0][0] for x in fin if x['stm']]),'n stmts mean',np.mean([len(x['stm']) for x in fin]))
print('finished content chars mean',np.mean([x['lencont'] for x in fin]))
json.dump(rows,open('work/bank_rows.json','w'))
