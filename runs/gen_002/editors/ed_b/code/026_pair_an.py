# Hypothesis: none (paired analysis tool: rank of gold, entropy, letter-prior shift between original and edited rows of a both pass).
import numpy as np, json, sys
k=sys.argv[1]
o={json.loads(l)['id']:json.loads(l) for l in open(f'work/fwd_{int(k):02d}/score_{k}_original.jsonl')}
e={json.loads(l)['id']:json.loads(l) for l in open(f'work/fwd_{int(k):02d}/score_{k}_edited.jsonl')}
ids=list(e)
def st(r):
    p=np.array(r['probs']); p=p[p>0]; p=p/p.sum() if p.sum()>0 else p
    pr=np.array(r['probs']); rank=(pr>pr[r['answer']]).sum()
    return rank, -(p*np.log(p)).sum(), np.log(np.array(r['probs'])[r['answer']]+1e-9)
O=np.array([st(o[i]) for i in ids]); E=np.array([st(e[i]) for i in ids])
print('mean gold rank o %.3f e %.3f | entropy o %.3f e %.3f | mean rank diff %.3f +- %.3f'%(O[:,0].mean(),E[:,0].mean(),O[:,1].mean(),E[:,1].mean(),(E[:,0]-O[:,0]).mean(),(E[:,0]-O[:,0]).std()/np.sqrt(len(ids))))
# mean log-prob change by original correctness
c=np.array([o[i]['correct'] for i in ids])
print('dlogp on orig-correct %.3f orig-wrong %.3f'%((E[c==1,2]-O[c==1,2]).mean(),(E[c==0,2]-O[c==0,2]).mean()))
