# Hypothesis: one gate can cover both </think> and the following '"The' (2-host design) with a usable CV margin.
import json, numpy as np, torch
from wsp.editor import *
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
pos=json.load(open('work/p2_targetpos.json'))
X=torch.from_numpy(np.concatenate([np.asarray(X2,dtype=np.float32),np.asarray(X1,dtype=np.float32)]))
textid=np.load('work/textid.npy'); excl=np.load('work/excl.npy'); ids_all=np.load('work/ids_all.npy'); N=X.shape[0]
P={'think':np.array(pos['think']),'qthe':np.array(pos['qthe']),'pe':np.array(pos['pe'])}
isP=np.zeros(N,bool)
for k in P: isP[P[k]]=True
neg=~isP&~excl
Pi=np.concatenate([P['think'],P['qthe']]); fold=textid[Pi]%2
for f in [0,1]:
    trn=neg&(textid%2!=f); Xn=X[torch.from_numpy(trn)][::3]; mN=Xn.mean(0); C=torch.cov(Xn.T); C=C+0.1*C.diag().mean()*torch.eye(2048)
    dv=torch.linalg.solve(C,X[Pi[fold!=f]].mean(0)-mN); s=(X-mN)@dv; tem=neg&(textid%2==f); m,sd=float(s[tem].mean()),float(s[tem].std())
    zt=(s[P['think'][textid[P['think']]%2==f]]-m)/sd; zq=(s[P['qthe'][textid[P['qthe']]%2==f]]-m)/sd; zn=(s[tem]-m)/sd; zp=(s[P['pe']]-m)/sd
    print(f'fold {f}: think min {float(zt.min()):.1f} qthe min {float(zq.min()):.1f} neg max {float(zn.max()):.1f} pe max {float(zp.max()):.1f}')
