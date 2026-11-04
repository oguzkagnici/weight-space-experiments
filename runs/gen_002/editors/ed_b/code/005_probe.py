# Hypothesis: residual stream at option-END tokens (what L22.H12 attends to) carries a correct-vs-incorrect signal before L22 that H12's attention under-uses.
import numpy as np, json
from sklearn.linear_model import LogisticRegression
recs=json.load(open('work/recs1.json'))
A=np.load('work/fwd_01/attn_last_L22.npy')
nq=len(recs); gold=np.array([r['gold'] for r in recs]); pred=np.array([r['pred'] for r in recs])
# H12 attention per option (end token) and span
for h in [12,15,13,14,11]:
    pe=[];ps=[]
    for r in recs:
        a=A[r['start']:r['start']+r['n'],h]
        pe.append(int(np.argmax([a[e] for e in r['ends']])))
        ps.append(int(np.argmax([a[p:e+1].sum() for p,e in zip(r['pos'],r['ends'])])))
    pe=np.array(pe); ps=np.array(ps)
    print(f'H{h}: argmax-end==gold {np.mean(pe==gold):.3f} ==pred {np.mean(pe==pred):.3f} | span==gold {np.mean(ps==gold):.3f}')
print('model acc',np.mean(pred==gold))
# probes
folds=np.arange(nq)%5
for lay in [14,18,20,22]:
    H=np.load(f'work/fwd_01/h{lay}.npy',mmap_mode='r')
    for kind in ['end','let']:
        X=[];y=[];g=[]
        for qi,r in enumerate(recs):
            P=r['ends'] if kind=='end' else r['pos']
            v=np.asarray(H[[r['start']+p for p in P]],dtype=np.float64)
            v=v/np.sqrt((v**2).mean(1,keepdims=True))   # rms normalize
            v=v-v.mean(0,keepdims=True)   # center within question
            X.append(v); y+= [int(L==r['gold']) for L in range(r['k'])]; g+=[qi]*r['k']
        X=np.concatenate(X); y=np.array(y); g=np.array(g)
        res={}
        for C in [0.001,0.01,0.1]:
            hit=[]; hitdm=[]
            for f in range(5):
                tr=folds[g]!=f; te=~tr
                clf=LogisticRegression(C=C,max_iter=2000).fit(X[tr],y[tr])
                d=X[tr][y[tr]==1].mean(0)-X[tr][y[tr]==0].mean(0)
                s=clf.decision_function(X[te]); sd=X[te]@d
                for qi in np.unique(g[te]):
                    m=g[te]==qi; hit.append(np.argmax(s[m])==recs[qi]['gold']); hitdm.append(np.argmax(sd[m])==recs[qi]['gold'])
            res[C]=(np.mean(hit),np.mean(hitdm))
        print(f'h{lay} {kind}: '+' '.join(f'C{C}: lr {a:.3f} dm {b:.3f}' for C,(a,b) in res.items()))
