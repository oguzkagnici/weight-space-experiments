# Hypothesis: none (pool paired holdout results over HOLD[0:512] for X1 (passes 6,9) and X0 (passes 7,10)).
import json, numpy as np
def load(k,m): return {json.loads(l)['id']:json.loads(l) for l in open(f'work/fwd_{k:02d}/score_{k}_{m}.jsonl')}
for nm,ks in [('X1',[6,9]),('X0',[7,10])]:
    d=[];lp=[];po=[]
    for k in ks:
        o=load(k,'original'); e=load(k,'edited')
        for i in e: d.append(e[i]['correct']-o[i]['correct']); lp.append(e[i]['logp_correct']-o[i]['logp_correct']); po.append(o[i]['correct'])
    d=np.array(d); lp=np.array(lp); n=len(d)
    print(f'{nm}: n {n} parent acc {100*np.mean(po):.2f} acc diff {100*d.mean():+.2f} SE {100*d.std(ddof=1)/np.sqrt(n):.2f} w->r {(d>0).sum()} r->w {(d<0).sum()} dlogp {lp.mean():+.3f} SE {lp.std(ddof=1)/np.sqrt(n):.3f}')
