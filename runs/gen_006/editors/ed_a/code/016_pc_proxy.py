# Hypothesis: removing the top principal components of thinking-token activations from the mean direction (bias proxy) and
# from the gate directions shrinks their thinking-token variance, giving a near-constant proxy and wide gate margins.
import json, numpy as np
D='work/fwd_01/'
off=np.load(D+'offsets.npy'); cpos=json.load(open('work/p1_closepos.json')); n=json.load(open(D+'meta.json'))['n_tokens']
X=np.load(D+'in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
# fit PCs on even texts' thinking, evaluate on odd texts' thinking (held out)
tr=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(0,13,2)])
te=np.concatenate([np.arange(off[j],off[j]+cpos[j]) for j in range(1,13,2)])
Xtr=np.asarray(X[tr],dtype=np.float32); Xte=np.asarray(X[te],dtype=np.float32)
mu=Xtr.mean(0); C=np.cov((Xtr-mu).T); ev,U=np.linalg.eigh(C); U=U[:,::-1]
natc=np.asarray(X[[off[j]+cpos[j]+k for j in range(13) for k in range(1,40) if cpos[j]+k<n[j]]],dtype=np.float32)
def T(k): return np.asarray(X[[off[j]+cpos[j]+k for j in range(13,26)]],dtype=np.float32)
for K in (0,20,50,100,200,400,800):
    P=U[:,:K]
    b=mu-P@(P.T@mu); b/=np.linalg.norm(b)
    pb=Xte@b
    out=f'K={K}: proxy CV(test) {pb.std()/pb.mean():.3f} mean {pb.mean():.2f};'
    for k in (0,1,6):
        Tk=T(k); d=Tk.mean(0)-mu; d=d-P@(P.T@d)
        s_te=Xte@d; s_T=Tk@d; s_N=natc@d
        # normalised by the proxy: score relative = s - t*pb with t chosen at midpoint in proxy units
        z=lambda s:(s-s_te.mean())/s_te.std()
        out+=f' p{k}: test max z {z(s_te).max():.1f}, target min z {z(s_T).min():.1f}, nat max z {z(s_N).max():.1f};'
    print(out)
