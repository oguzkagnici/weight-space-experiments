# Hypothesis: the option-end correctness signal is complementary to H12's attention (ensemble beats H12 attention / model), i.e. the key path under-reads it.
import numpy as np, json
recs=json.load(open('work/recs1.json')); A=np.load('work/fwd_01/attn_last_L22.npy')
nq=len(recs); gold=np.array([r['gold'] for r in recs]); pred=np.array([r['pred'] for r in recs])
folds=np.arange(nq)%5
att=[np.log(A[r['start']:r['start']+r['n'],12][r['ends']]+1e-9) for r in recs]
lg=[np.array(r['lg']) for r in recs]
for lay in [14,18,20,22]:
    H=np.load(f'work/fwd_01/h{lay}.npy',mmap_mode='r')
    X=[];y=[];g=[]
    for qi,r in enumerate(recs):
        v=np.asarray(H[[r['start']+p for p in r['ends']]],dtype=np.float64)
        v=v/np.sqrt((v**2).mean(1,keepdims=True)); v=v-v.mean(0,keepdims=True)
        X.append(v); y+=[int(L==r['gold']) for L in range(r['k'])]; g+=[qi]*r['k']
    X=np.concatenate(X); y=np.array(y); g=np.array(g)
    sc=np.zeros(len(y))
    for f in range(5):
        tr=folds[g]!=f; te=~tr
        d=X[tr][y[tr]==1].mean(0)-X[tr][y[tr]==0].mean(0)
        sc[te]=X[te]@d/ (np.std(X[tr]@d)+1e-9)
    ps=[sc[g==qi] for qi in range(nq)]
    out=[]
    for b in [0,0.25,0.5,1,2,4]:
        a1=np.mean([np.argmax(att[i]+b*ps[i])==gold[i] for i in range(nq)])
        a2=np.mean([np.argmax(lg[i]+b*ps[i])==gold[i] for i in range(nq)])
        out.append(f'b{b}: att {a1:.3f} logit {a2:.3f}')
    wr=pred!=gold
    print(f'h{lay}', ' | '.join(out), f'| probe acc on model-wrong {np.mean([np.argmax(ps[i])==gold[i] for i in np.where(wr)[0]]):.3f} on right {np.mean([np.argmax(ps[i])==gold[i] for i in np.where(~wr)[0]]):.3f}')
