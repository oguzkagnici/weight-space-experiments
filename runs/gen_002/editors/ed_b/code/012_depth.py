# Hypothesis: the option-end correctness signal is written in a specific depth block (layers 4..16); identify it by CV probe accuracy per layer and by transfer of the pass-1 h14 direction.
from wsp.editor import *
import numpy as np, json
qs=questions(); sel=json.load(open('work/p2_qids.json')); off=np.load('work/fwd_02/offsets_tok.npy')
OPT=[17175,20370,16384,21639,32642,25413,21010,28899,28687,44007]
recs=[]
for i,q in enumerate(sel):
    ids=tok.encode(render(q)); n=len(ids); assert n==off[i+1]-off[i]
    k=len(qs[q]['options']); pos=[None]*k; j=n-1
    for L in range(k-1,-1,-1):
        while ids[j]!=OPT[L]: j-=1
        pos[L]=j; j-=1
    ans=[t for t in range(pos[-1],n) if tok.decode([ids[t]]).startswith('Answer')][0]
    ends=[pos[L+1]-1 for L in range(k-1)]+[ans-1]
    recs.append(dict(q=q,k=k,gold=qs[q]['answer_index'],start=int(off[i]),n=n,pos=pos,ends=ends,cat=qs[q]['category']))
A=np.load('work/fwd_02/attn_last_L22.npy')
for r in recs: r['h12']=int(np.argmax(A[r['start']:r['start']+r['n'],12][r['ends']]))
json.dump(recs,open('work/recs2.json','w'))
gold=np.array([r['gold'] for r in recs]); h12=np.array([r['h12'] for r in recs]); nq=len(recs)
print('H12 argmax acc on pass-2 set',np.mean(h12==gold))
idx=np.concatenate([[r['start']+e for e in r['ends']] for r in recs]); g=np.concatenate([[i]*r['k'] for i,r in enumerate(recs)])
y=np.concatenate([[int(L==r['gold']) for L in range(r['k'])] for r in recs])
np.savez('work/p2_index.npz',idx=idx,g=g,y=y)
# pass-1 direction at h14
p1=np.load('work/p1_feats.npz'); r1=json.load(open('work/recs1.json'))
g1=np.concatenate([[i]*r['k'] for i,r in enumerate(r1)]); y1=np.concatenate([[int(L==r['gold']) for L in range(r['k'])] for r in r1])
def prep(X,g):
    X=X/np.sqrt((X**2).mean(1,keepdims=True))
    for qi in np.unique(g): X[g==qi]-=X[g==qi].mean(0)
    return X
X1=prep(p1['end14'].astype(np.float64),g1); d14=X1[y1==1].mean(0)-X1[y1==0].mean(0); np.save('work/d14_p1.npy',d14)
folds=np.arange(nq)%5
for lay in [4,8,10,12,14,16]:
    H=np.load(f'work/fwd_02/h{lay}.npy',mmap_mode='r'); X=prep(np.asarray(H[idx],dtype=np.float64),g)
    np.save(f'work/p2_end{lay}.npy',np.asarray(H[idx],dtype=np.float32))
    sc=np.zeros(len(y))
    for f in range(5):
        tr=folds[g]!=f; te=~tr; d=X[tr][y[tr]==1].mean(0)-X[tr][y[tr]==0].mean(0); sc[te]=X[te]@d
    hit=np.array([np.argmax(sc[g==qi])==gold[qi] for qi in range(nq)])
    s1=X@d14; hit1=np.array([np.argmax(s1[g==qi])==gold[qi] for qi in range(nq)])
    agr=np.array([np.argmax(sc[g==qi])==h12[qi] for qi in range(nq)])
    # separation in sd units
    sep=(s1[y==1].mean()-s1[y==0].mean())/s1[y==0].std()
    print(f'h{lay}: CV-DM acc {hit.mean():.3f} (agree H12 {agr.mean():.3f}) | p1-d14 transfer acc {hit1.mean():.3f} sep {sep:.2f}')
