# Hypothesis: (fix of 023; v4: gate zero-crossing midway between in-sample neg max and pos min, |g|=20 at both) choosing the gate threshold on the actual gate (incl. bias-proxy noise) to maximise the pos/neg ratio, and scaling the write column into Q8 block-max saturation, gives off-target gate <= -14 (exact-zero leak) and a write direction close to the ideal token direction.
import json, numpy as np, torch
from wsp.editor import *
LAM=0.1; GNEG=-14.0; KAPPA=4.0
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
pos=json.load(open('work/p2_targetpos.json'))
X=torch.from_numpy(np.concatenate([np.asarray(X2,dtype=np.float32),np.asarray(X1,dtype=np.float32)]))
textid=np.load('work/textid.npy'); excl=np.load('work/excl.npy'); N=X.shape[0]
H2=np.load('work/fwd_02/mlp23.npy',mmap_mode='r'); H1=np.load('work/fwd_01/mlp23.npy',mmap_mode='r')
P={'think':np.array(pos['think']),'qthe':np.array(pos['qthe']),'pe':np.array(pos['pe'])}
isP=np.zeros(N,bool)
for k in P: isP[P[k]]=True
neg=~isP&~excl; negt=torch.from_numpy(neg)
bf=lambda t:t.to(torch.bfloat16).float()
W1=load_original('model.layers.23.feed_forward.w1.weight'); W3=load_original('model.layers.23.feed_forward.w3.weight'); W2b=bf(load_original('model.layers.23.feed_forward.w2.weight'))
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ti=np.load('work/fwd_02/topk_ids.npy')
w=torch.load('work/l23_biasproxy.pt')['w']; r=X@w; Wbar=float(r[negt].mean()); r=r/Wbar
Xn=X[negt][::3]; mN=Xn.mean(0); C=torch.cov(Xn.T); C=C+LAM*C.diag().mean()*torch.eye(2048)
UNITS=[('think',3401,1228,14225),('qthe',3793,6986,5141),('pe',881,331,124900)]
W1n,W3n,W2n=W1.clone(),W3.clone(),W2b.clone()
silu=torch.nn.functional.silu
for name,j,k,tgt in UNITS:
    W2n[:,k]=bf(W2b[:,k]+W2b[:,j])
    d=torch.linalg.solve(C,X[P[name]].mean(0)-mN); s=X@d
    best=None
    for tau in np.linspace(float(s[negt].mean()),float(s[P[name]].min()),400):
        z=s-tau*r; sd=float(z[negt].std()); mn=float(z[negt].max()); mp=float(z[P[name]].min())
        if best is None or (mp-mn)/sd>best[0]: best=((mp-mn)/sd,tau,mn,mp,sd)
    sep,tau,mn,mp,sd=best
    G0=s-tau*r                           # combined score; the constant T must also go through the proxy r
    lo,hi=-1e4,1e4
    def fT(T):
        z=G0-T*r; return float(z[negt].max()), float(z[P[name]].min())
    Ts=np.linspace(-3*abs(float(G0[negt].max()))-10,3*abs(float(G0[P[name]].min()))+10,4001)
    bestT=None
    for T in Ts:
        mn_,mp_=fT(T)
        if mn_<0<mp_:
            asym=abs(mp_+mn_)
            if bestT is None or asym<bestT[0]: bestT=(asym,T,mn_,mp_)
    _,T,mn_,mp_=bestT; a=20.0/(-mn_)
    row=a*(d - (tau+T)*w/Wbar); g=X@row
    up=KAPPA*w/Wbar; u=X@up; h=silu(g)*u
    b=(j//32)*32; others=[i for i in range(b,b+32) if i!=j]
    amax=torch.from_numpy(np.abs(np.concatenate([np.asarray(H2[:,others],dtype=np.float32),np.asarray(H1[:,others],dtype=np.float32)])).max(1))
    lr=h.abs()/(amax/254)
    zz=(g-float(g[negt].mean()))/float(g[negt].std())
    print(f'{name}: host {j}: gap {sep:.1f} neg-sd | gate neg max {float(g[negt].max()):.1f} (z {float(zz[negt].max()):.1f}) | gate=0 at z {float(-g[negt].mean()/g[negt].std()):.1f} | pos min {float(g[P[name]].min()):.1f} med {float(g[P[name]].median()):.1f} | other targets max {max(float(g[P[o]].max()) for o in P if o!=name):.1f} | leak: max ratio {float(lr[negt].max()):.4f}, nonzero-Q8 {int((lr[negt]>=0.5).sum())}')
    W1n[j]=row; W3n[j]=up
    comp=set()
    for p in P[name]:
        if p < X2.shape[0]: comp.update(int(t) for t in ti[p][:10])
    comp.discard(tgt); comp=torch.tensor(sorted(comp))
    dl=gf*(E[tgt]-E[comp].mean(0)); col=dl/gpf
    blk=W2b[:,b:b+32].abs(); oldmax=blk.max(1).values; othmax=torch.cat([blk[:,:j-b],blk[:,j-b+1:]],1).max(1).values
    need=othmax<oldmax
    bestc=None
    for lam in np.logspace(-2,3,60):
        c=torch.clamp(col/col.abs().max()*lam*oldmax.max(),-oldmax,oldmax)
        c=torch.where(need,torch.sign(col)*oldmax,c)
        cs=float((gpf*c)@dl/((gpf*c).norm()*dl.norm()))
        if bestc is None or cs>bestc[0]: bestc=(cs,lam,c)
    cs,lam,c=bestc; c=bf(c)
    assert bool((torch.maximum(c.abs(),othmax)==oldmax).all())
    W2n[:,j]=c
    Ht=np.asarray(H2[P[name][P[name]<X2.shape[0]]],dtype=np.float32); o=torch.from_numpy(Ht)@W2b.T
    hp=h[P[name]]
    print(f'   write: cos(realised, ideal) {cs:.3f} col norm {float(c.norm()):.2f} | |o| at targets med {float(o.norm(dim=1).median()):.1f} | |h*col| min {float(hp.min()*c.norm()):.0f} med {float(hp.median()*c.norm()):.0f}')
    # logit-lens of the realised write (post-norm direction) over the vocab: rank of target
    lv=(gf*gpf*c)@E.T; print('   target rank in write direction:',int((lv>lv[tgt]).sum()),' top ids',torch.topk(lv,5).indices.tolist())
save_edit('model.layers.23.feed_forward.w1.weight',W1n); save_edit('model.layers.23.feed_forward.w3.weight',W3n); save_edit('model.layers.23.feed_forward.w2.weight',W2n)
