# Hypothesis: correctness info the model fails to use exists in option-span means (not just end tokens): a span probe beats chance on model-wrong questions.
import numpy as np, json
from sklearn.linear_model import LogisticRegression
recs=json.load(open('work/recs1.json')); nq=len(recs); folds=np.arange(nq)%5
gold=np.array([r['gold'] for r in recs]); pred=np.array([r['pred'] for r in recs]); wr=pred!=gold
for lay in [14,18,20,22]:
    H=np.load(f'work/fwd_01/h{lay}.npy',mmap_mode='r')
    X=[];y=[];g=[]
    for qi,r in enumerate(recs):
        V=np.asarray(H[r['start']:r['start']+r['n']],dtype=np.float64); V=V/np.sqrt((V**2).mean(1,keepdims=True))
        v=np.stack([V[p+2:e+1].mean(0) if e>=p+2 else V[e] for p,e in zip(r['pos'],r['ends'])]); v-=v.mean(0)
        X.append(v); y+=[int(L==r['gold']) for L in range(r['k'])]; g+=[qi]*r['k']
    X=np.concatenate(X); y=np.array(y); g=np.array(g)
    for name in ['dm','lr']:
        sc=np.zeros(len(y))
        for f in range(5):
            tr=folds[g]!=f; te=~tr
            if name=='dm': d=X[tr][y[tr]==1].mean(0)-X[tr][y[tr]==0].mean(0); sc[te]=X[te]@d
            else: sc[te]=LogisticRegression(C=0.003,max_iter=3000).fit(X[tr],y[tr]).decision_function(X[te])
        hit=np.array([np.argmax(sc[g==qi])==gold[qi] for qi in range(nq)])
        print(f'h{lay} span {name}: all {hit.mean():.3f} model-wrong {hit[wr].mean():.3f} model-right {hit[~wr].mean():.3f}')
