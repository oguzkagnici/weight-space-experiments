# Hypothesis: a gate direction built from means (churn breaks vs other breaks, plus churn breaks vs all tokens) can fire on most churn breaks while firing on almost no ordinary tokens and few finished-trace breaks (cross-validated by trace).
import json, numpy as np
from wsp.editor import *
D='work/fwd_01'
recs=json.load(open('work/009_breaks.json'))
offs=np.load(f'{D}/offsets.npy'); ids_list=json.load(open('work/req1_ids.json'))
rows={json.loads(l)['id']:json.loads(l) for l in open('/home/user/weight-space-experiments/runs/shared/trace_bank_base_cap3072_det1.jsonl')}
pch=np.load('work/010_pch.npy')
rowsI=np.array([r['row'] for r in recs]); cls=np.array([r['cls'] for r in recs]); cut=np.array([r['cut'] for r in recs]); ti=np.array([r['ti'] for r in recs])
N=int(offs[-1]); tok_ti=np.repeat(np.arange(len(ids_list)),np.diff(offs))
isbrk=np.zeros(N,bool); isbrk[rowsI]=True
tcut=np.array([rows[q]['cut'] for q in ids_list])[tok_ti]
rng=np.random.default_rng(1); samp=np.sort(rng.choice(np.where(~isbrk)[0],30000,replace=False))
for L in (20,23):
    X=np.load(f'{D}/in_model.layers.{L}.feed_forward.w1.npy',mmap_mode='r')
    Xb=np.asarray(X[rowsI],np.float32); Xs=np.asarray(X[samp],np.float32); sti=tok_ti[samp]
    for name in ('g1','g2','g1+g2'):
        res=[]
        for f in range(4):
            trb=(ti%4!=f); trs=(sti%4!=f)
            ch=Xb[trb&(cls=='churn')].mean(0)
            g1=ch-Xb[trb&(cls=='other')].mean(0); g2=ch-Xs[trs].mean(0)
            g={'g1':g1/np.linalg.norm(g1),'g2':g2/np.linalg.norm(g2),'g1+g2':g1/np.linalg.norm(g1)+g2/np.linalg.norm(g2)}[name]
            # threshold: fires on 60% of TRAIN churn breaks
            th=np.percentile(Xb[trb&(cls=='churn')]@g,40)
            tb=~trb; ts=~trs
            sb=Xb[tb]@g; ss=Xs[ts]@g
            res.append(((sb[cls[tb]=='churn']>th).mean(), (ss>th).mean(), (sb[(cls[tb]!='churn')&(cut[tb]==0)]>th).mean(),(sb[(cls[tb]!='churn')&(cut[tb]==1)]>th).mean(),(sb[cls[tb]=='commit']>th).mean()))
        r=np.array(res).mean(0)
        print(f'L{L} {name}: heldout fire rate: churn breaks {r[0]:.2f} | non-break tokens {r[1]*100:.2f}% | non-churn breaks fin {r[2]:.3f} cut {r[3]:.3f} | commit breaks {r[4]:.3f}')
