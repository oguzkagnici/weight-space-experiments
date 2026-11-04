# Hypothesis: none (pool paired differences of v2 over HOLDOUT[0:768] from passes 5,6,9; v1 vs v2 on HOLDOUT[0:256]).
import numpy as np, json
def d(k):
    o={r['id']:r for r in map(json.loads,open(f'work/fwd_{k:02d}/score_{k}_original.jsonl'))}
    e={r['id']:r for r in map(json.loads,open(f'work/fwd_{k:02d}/score_{k}_edited.jsonl'))}
    return np.array([e[i]['correct']-o[i]['correct'] for i in o]), np.array([e[i]['logp_correct']-o[i]['logp_correct'] for i in o])
A=[d(k) for k in [5,6,9]]; a=np.concatenate([x[0] for x in A]); l=np.concatenate([x[1] for x in A])
print('v2 HOLDOUT n',len(a),'dacc %.2f +- %.2f'%(100*a.mean(),100*a.std(ddof=1)/np.sqrt(len(a))),'dlogp %.3f +- %.3f'%(l.mean(),l.std(ddof=1)/np.sqrt(len(l))))
for k in [3,4]:
    a,l=d(k); print('S pass',k,'dacc %.2f +- %.2f'%(100*a.mean(),100*a.std(ddof=1)/np.sqrt(len(a))))
