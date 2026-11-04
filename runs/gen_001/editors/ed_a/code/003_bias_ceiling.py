# Hypothesis: removing a constant per-letter logit bias at the output would raise accuracy; estimate the ceiling offline from recorded letter probs (2-fold CV).
import numpy as np, json
from wsp.editor import questions
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions()
nopt=np.array([len(Q[r['id']]['options']) for r in rows]); ans=np.array([r['answer'] for r in rows])
P=np.array([r['probs']+[0.0]*(10-len(r['probs'])) for r in rows])
L=np.log(P+1e-12); mask=np.arange(10)[None]<nopt[:,None]
L=np.where(mask,L,-1e9)
def bias(idx):
    m=idx[nopt[idx]==10]; Lc=L[m]-L[m].mean(1,keepdims=True); return Lc.mean(0)
rng=np.random.default_rng(0); perm=rng.permutation(len(rows)); A,B=perm[:256],perm[256:]
print('base acc',(L.argmax(1)==ans).mean())
for c in [0.25,0.5,0.75,1.0]:
    acc=[]
    for tr,te in [(A,B),(B,A)]:
        b=bias(tr); Lb=L[te]-c*b; acc.append((Lb.argmax(1)==ans[te]).mean())
    lpc=[]
    print(c, 'cv acc',np.mean(acc))
b=bias(np.arange(len(rows))); print('bias',np.round(b,2))
Lb=L-b; print('pred counts debiased',np.bincount(Lb.argmax(1),minlength=10))
