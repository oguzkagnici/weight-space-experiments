# Hypothesis: at churn breaks the model's alternatives to the churn opener include commit openers (Thus/Hence/Given/I) with non-trivial mass, so suppressing churn would redirect toward them.
import json, numpy as np, collections
from wsp.editor import *
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json'))
topi=np.load(f'{D}/topk_ids.npy'); topv=np.load(f'{D}/topk_logprobs.npy')
pch=np.load('work/010_pch.npy')
for name,sel in (('churn breaks (cut)',[i for i,r in enumerate(recs) if r['cls']=='churn' and r['cut']]),
                 ('high-pch breaks pch>.2 (all)',[i for i,r in enumerate(recs) if pch[i]>.2]),
                 ('commit breaks (fin)',[i for i,r in enumerate(recs) if r['cls']=='commit' and not r['cut']])):
    acc=collections.Counter()
    for i in sel:
        for t,v in zip(topi[recs[i]['row']],topv[recs[i]['row']]): acc[int(t)]+=np.exp(v)/len(sel)
    print(name,len(sel),[(tok.decode([t]).strip(),round(p,3)) for t,p in acc.most_common(16)])
print('n high-pch breaks by cut:',collections.Counter(recs[i]['cut'] for i in range(len(recs)) if pch[i]>.2))
print('traces with >=1 high-pch break: cut',len(set(r['qid'] for i,r in enumerate(recs) if pch[i]>.2 and r['cut'])),'fin',len(set(r['qid'] for i,r in enumerate(recs) if pch[i]>.2 and not r['cut'])))
