# Hypothesis: with top-200 thinking PCs projected out, the p0 and p6 gates keep a wide margin in leave-one-out tests; the p1
# position ("The" after </think>) becomes separable when combined with a "current token is The" direction.
import json, numpy as np
from wsp.editor import tok
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json')); n=json.load(open(D+'meta.json'))['n_tokens']
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
texts=json.load(open('work/req_p1.json'))['texts']
tr=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(0,13,2)])
te=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(1,13,2)])
Xtr=np.asarray(X[tr],dtype=np.float32); Xte=np.asarray(X[te],dtype=np.float32)
mu=Xtr.mean(0); C=np.cov((Xtr-mu).T); ev,U=np.linalg.eigh(C); P=U[:,::-1][:,:200]
b=mu-P@(P.T@mu); b/=np.linalg.norm(b)
natidx=[off[j]+cpos[j]+k for j in range(13) for k in range(1,40) if cpos[j]+k<n[j]]
natc=np.asarray(X[natidx],dtype=np.float32)
def T(k): return np.asarray(X[[off[j]+cpos[j]+k for j in range(13,26)]],dtype=np.float32)
for k in (0,6):
    Tk=T(k); loo=[]
    for i in range(13):
        d=np.delete(Tk,i,0).mean(0)-mu; d=d-P@(P.T@d)
        s=Xte@d; loo.append(((Tk[i]@d)-s.mean())/s.std())
    d=Tk.mean(0)-mu; d=d-P@(P.T@d); s=Xte@d; zN=(natc@d-s.mean())/s.std()
    print(f'p{k}: LOO target z min {min(loo):.1f} median {np.median(loo):.1f}; test-thinking max z {((s-s.mean())/s.std()).max():.1f}; natural content max z {zN.max():.1f}')
    top=np.argsort(-zN)[:5]
    for t in top:
        g=natidx[t]; j=int(np.searchsorted(off,g,side='right')-1); kk=g-off[j]-cpos[j]
        ids=tok(texts[j],add_special_tokens=False)['input_ids']
        print('    nat z %.1f text %d p%d ctx %r'%(zN[t],j,kk,tok.decode(ids[cpos[j]:cpos[j]+kk+1])[-60:]))
