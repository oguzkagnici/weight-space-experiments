# Hypothesis: L22.H12's attention over option ends has a position (option-index) prior; removing it (CV) would make H12 select the gold option more often.
import numpy as np, json
recs=json.load(open('work/recs1.json')); A=np.load('work/fwd_01/attn_last_L22.npy')
R=[r for r in recs if r['k']==10]; n=len(R); gold=np.array([r['gold'] for r in R]); pred=np.array([r['pred'] for r in R])
for h in [12,15,14,13,11]:
    la=np.array([np.log(A[r['start']:r['start']+r['n'],h][r['ends']]+1e-9) for r in R]); la-=la.mean(1,keepdims=True)
    mask=np.ones_like(la,bool); mask[np.arange(n),gold]=False
    bias_all=la.mean(0); bias_ng=np.array([la[mask[:,i],i].mean() for i in range(10)])
    folds=np.arange(n)%5; hit=0
    for f in range(5):
        b=la[folds!=f].mean(0); te=folds==f
        hit+=np.sum(np.argmax(la[te]-b,1)==gold[te])
    pm=np.exp(A[0:1,0]*0)  # dummy
    print(f'H{h} raw acc {np.mean(la.argmax(1)==gold):.3f} debiased(CV) {hit/n:.3f}  bias(non-gold) '+' '.join(f'{v:+.2f}' for v in bias_ng))
    print('   argmax counts', np.bincount(la.argmax(1),minlength=10), ' gold counts', np.bincount(gold,minlength=10))
la=np.array([np.log(A[r['start']:r['start']+r['n'],12][r['ends']]+1e-9) for r in R])
# total attention mass on option ends, and rank of gold among H12 attention when wrong
rk=np.array([(la[i]>la[i,gold[i]]).sum() for i in range(n)])
print('H12 gold rank dist',np.bincount(rk,minlength=10))
lg=np.array([r['lg'] for r in R]); rk2=np.array([(lg[i]>lg[i,gold[i]]).sum() for i in range(n)])
print('final logit gold rank dist',np.bincount(rk2,minlength=10), 'pred counts',np.bincount(pred,minlength=10))
lgc=lg-lg.mean(1,keepdims=True); print('mean centered final logits',np.round(lgc.mean(0),2))
