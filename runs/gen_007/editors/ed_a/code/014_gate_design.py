# Hypothesis: at the L23 MLP input, </think> positions (H1) and the post-think '"The' / ')".' positions (H2) are linearly separable from ALL thinking/prompt tokens with a margin wide enough for a gate that is <-20 off-target (silu leak quantises to exactly 0) and >0 on target.
import json, numpy as np, torch
from wsp.editor import *
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
pos=json.load(open('work/p2_targetpos.json'))
X=torch.from_numpy(np.concatenate([np.asarray(X2,dtype=np.float32),np.asarray(X1,dtype=np.float32)]))
T2=X2.shape[0]; N=X.shape[0]
# pass-1 texts that contain the original visible answers: exclude everything after </think> there (not seen by edited model)
d1=json.load(open('work/p1_texts.json')); off1=np.load('work/fwd_01/offsets.npy')
excl=np.zeros(N,bool)
for i,t in enumerate(d1['texts']):
    ids=tok(t,add_special_tokens=False).input_ids
    if 124902 in ids:
        p=ids.index(124902); excl[T2+off1[i]+p:T2+off1[i+1]]=True   # drop </think> (dup of pass 2) and the original answer
# pass-2 texts: drop the post-think answer tokens except targets; they are separate negatives
d2=json.load(open('work/p2_texts.json')); off2=np.load('work/fwd_02/offsets.npy')
ans=np.zeros(N,bool)
for i in range(len(d2['texts'])):
    if d2['kind'][i]=='cut': continue
for p in pos['think']: ans[p+1:p+8]=True
P1=np.array(pos['think']); P2=np.array(pos['qthe']+pos['pe'])
isP=np.zeros(N,bool); isP[P1]=True; isP[P2]=True
neg=~isP & ~excl
print('N',N,'neg',neg.sum(),'P1',len(P1),'P2',len(P2))
mu=X[neg].mean(0); sd=X[neg].std(0)
cv=(sd/mu.abs()); o=torch.argsort(cv)[:5]
print('most constant dims',[(int(i),round(float(mu[i]),2),round(float(cv[i]),3)) for i in o])
# bias proxy: mean direction
v=mu/mu.norm(); pv=X@v; print('bias proxy proj mean',float(pv[neg].mean()),'cv',float(pv[neg].std()/pv[neg].mean()))
Xn=X[neg][::4]-mu
U,S,Vt=torch.linalg.svd(Xn,full_matrices=False)
torch.save({'mu':mu,'Vt':Vt[:400],'S':S[:400]},'work/l23_pcs.pt')
def evaluate(Pidx,name,K):
    dvec=X[Pidx].mean(0)-mu
    if K>0:
        B=Vt[:K]; dvec=dvec-B.T@(B@dvec)
    s=(X-mu)@dvec
    sn=s[neg]; m,sdv=float(sn.mean()),float(sn.std())
    z=lambda a:(a-m)/sdv
    # held-out positives: split by halves
    h=len(Pidx)//2
    print(f'{name} K={K}: pos z min {float(z(s[Pidx]).min()):.1f} med {float(z(s[Pidx]).median()):.1f} | neg z max {float(z(sn).max()):.1f}, 99.99% {float(torch.quantile(z(sn[::7]),0.9999)):.1f} | other-answer-tokens z max {float(z(s[torch.from_numpy(ans&~isP)]).max()):.1f}')
    return dvec
for K in [0,50,200]:
    evaluate(torch.from_numpy(P1),'H1 </think>',K); evaluate(torch.from_numpy(P2),'H2 qthe+pe',K)
