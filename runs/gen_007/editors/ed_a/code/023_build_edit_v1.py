# Hypothesis: three freed L23 duplicate slots (3401, 3793, 881; folded into twins 1228, 6986, 331) can host sharp LDA-gated units -- </think> -> '"The', post-think '"The' -> ' answer', post-think ')".' -> <|im_end|> -- whose leak quantises to exactly 0 on all recorded thinking tokens.
import json, numpy as np, torch
from wsp.editor import *
LAM=0.1; GNEG=-12.0       # gate value required at the in-sample max negative
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
pos=json.load(open('work/p2_targetpos.json'))
X=torch.from_numpy(np.concatenate([np.asarray(X2,dtype=np.float32),np.asarray(X1,dtype=np.float32)]))
textid=np.load('work/textid.npy'); excl=np.load('work/excl.npy'); ids_all=np.load('work/ids_all.npy'); N=X.shape[0]
H2=np.load('work/fwd_02/mlp23.npy',mmap_mode='r'); H1=np.load('work/fwd_01/mlp23.npy',mmap_mode='r')
P={'think':np.array(pos['think']),'qthe':np.array(pos['qthe']),'pe':np.array(pos['pe'])}
isP=np.zeros(N,bool)
for k in P: isP[P[k]]=True
neg=~isP&~excl; negt=torch.from_numpy(neg)
W1=load('model.layers.23.feed_forward.w1.weight'); W3=load('model.layers.23.feed_forward.w3.weight'); W2o=load('model.layers.23.feed_forward.w2.weight')
bf=lambda t:t.to(torch.bfloat16).float()
W2b=bf(W2o)
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ti=np.load('work/fwd_02/topk_ids.npy')
w=torch.load('work/l23_biasproxy.pt')['w']; pw=X@w; Wbar=float(pw[negt].mean())
Xn=X[negt][::3]; mN=Xn.mean(0); C=torch.cov(Xn.T); C=C+LAM*C.diag().mean()*torch.eye(2048)
UNITS=[('think',3401,1228,14225),('qthe',3793,6986,5141),('pe',881,331,124900)]
W1n,W3n,W2n=W1.clone(),W3.clone(),W2b.clone()
info={}
for name,j,k,tgt in UNITS:
    # fold j into twin k (identical units: h_j ~ h_k, w2_j ~ w2_k), j becomes free
    W2n[:,k]=bf(W2b[:,k]+W2b[:,j])
    d=torch.linalg.solve(C,X[P[name]].mean(0)-mN)
    s=X@d
    # gate g = a*s - b*(x.w)/Wbar ; choose a,b so that max over negatives = GNEG and the bias is applied through the proxy
    sn=s[negt]; sp=s[P[name]]
    # set threshold midway (in s units) between neg max and pos min, then scale
    tau=float(sn.max())+0.45*(float(sp.min())-float(sn.max()))
    a=GNEG/(float(sn.max())-tau)
    row=a*d - a*tau*w/Wbar
    g=X@row
    up=w/Wbar                       # up ~ 1 everywhere
    u=X@up
    hneg=torch.nn.functional.silu(g[negt])*u[negt]
    # quantisation check: block of j (others' max after freeing j)
    b=(j//32)*32; others=[i for i in range(b,b+32) if i!=j]
    Hall=np.concatenate([np.asarray(H2[:,others],dtype=np.float32),np.asarray(H1[:,others],dtype=np.float32)])
    amax=torch.from_numpy(np.abs(Hall).max(1))
    ratio=(torch.nn.functional.silu(g)*u).abs()/(amax/254)
    nzq=int((ratio[negt]>=0.5).sum())
    print(f'{name}: host {j} twin {k}: a={a:.3f} tau={tau:.1f} | gate neg max {float(g[negt].max()):.1f} p99.9 {float(torch.quantile(g[negt][::5],0.999)):.1f} | pos min {float(g[P[name]].min()):.1f} med {float(g[P[name]].median()):.1f} | other targets max {max(float(g[P[o]].max()) for o in P if o!=name):.1f} | neg tokens with nonzero-Q8 leak: {nzq} | max leak ratio {float(ratio[negt].max()):.3f}')
    W1n[j]=row; W3n[j]=up
    # write column: target minus mean of competitors seen at these positions (top-10)
    comp=set()
    for p in P[name]:
        if p < X2.shape[0]: comp.update(int(t) for t in ti[p][:10])
    comp.discard(tgt); comp=torch.tensor(sorted(comp))
    y=gf*(E[tgt]-E[comp].mean(0))
    col=y/gpf
    col=col/col.abs().max()
    # scale-preserving clip: entries <= old block absmax (others incl. old j); if old j was the unique max, set |entry| = old max
    blk=W2b[:,b:b+32].abs(); oldmax=blk.max(1).values; othmax=torch.cat([blk[:,:j-b],blk[:,j-b+1:]],1).max(1).values
    c=col*oldmax*1.0
    c=torch.clamp(c,-oldmax,oldmax)
    need=othmax<oldmax                 # j was the unique max in these rows
    c=torch.where(need, torch.sign(c+1e-12)*oldmax, c)
    c=bf(c)
    assert bool((torch.maximum(c.abs(),othmax)==oldmax).all()), 'scale changed'
    W2n[:,j]=c
    # dominance: |col|*h at targets vs |o| at targets
    Ht=np.concatenate([np.asarray(H2[P[name][P[name]<X2.shape[0]]],dtype=np.float32)]); o=torch.from_numpy(Ht)@W2b.T
    hp=torch.nn.functional.silu(g[P[name]])*u[P[name]]
    yl=(gf*gpf*c)@E[torch.tensor([tgt])].T
    print(f'   col norm {float(c.norm()):.2f} cos(realised, ideal) {float((gpf*c)@y/((gpf*c).norm()*y.norm())):.3f} | |o| at targets med {float(o.norm(dim=1).median()):.1f} | h at targets med {float(hp.median()):.1f} -> |h col| med {float(hp.median())*float(c.norm()):.1f}; rows forced to old max {int(need.sum())}; comps {len(comp)}')
save_edit('model.layers.23.feed_forward.w1.weight',W1n); save_edit('model.layers.23.feed_forward.w3.weight',W3n); save_edit('model.layers.23.feed_forward.w2.weight',W2n)
print(edited())
