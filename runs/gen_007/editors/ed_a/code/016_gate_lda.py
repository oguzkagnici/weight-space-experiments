# Hypothesis: a shrinkage-LDA direction (C+lam I)^-1 (mu_P - mu_N) separates the targets from all thinking tokens with a wider cross-validated margin than PC-projected mean differences, incl. a combined qthe+pe (H2) gate.
import json, numpy as np, torch
from wsp.editor import *
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
pos=json.load(open('work/p2_targetpos.json'))
X=torch.from_numpy(np.concatenate([np.asarray(X2,dtype=np.float32),np.asarray(X1,dtype=np.float32)]))
T2=X2.shape[0]; N=X.shape[0]
d1=json.load(open('work/p1_texts.json')); off1=np.load('work/fwd_01/offsets.npy'); off2=np.load('work/fwd_02/offsets.npy'); d2=json.load(open('work/p2_texts.json'))
ids_all=[]; textid=np.zeros(N,int)
for i,t in enumerate(d2['texts']): ids_all+=tok(t,add_special_tokens=False).input_ids; textid[off2[i]:off2[i+1]]=i
excl=np.zeros(N,bool)
for i,t in enumerate(d1['texts']):
    ids=tok(t,add_special_tokens=False).input_ids; ids_all+=ids; textid[T2+off1[i]:T2+off1[i+1]]=100+i
    if 124902 in ids: p=ids.index(124902); excl[T2+off1[i]+p:T2+off1[i+1]]=True
ids_all=np.array(ids_all); np.save('work/ids_all.npy',ids_all); np.save('work/textid.npy',textid); np.save('work/excl.npy',excl)
P={'think':np.array(pos['think']),'qthe':np.array(pos['qthe']),'pe':np.array(pos['pe'])}; P['H2']=np.concatenate([P['qthe'],P['pe']])
isP=np.zeros(N,bool)
for k in ['think','qthe','pe']: isP[P[k]]=True
neg=~isP&~excl
for lam in [0.1,0.01]:
  for name in ['think','qthe','pe','H2']:
    Pi=P[name]; fold=textid[Pi]%2; res=[]
    for f in [0,1]:
        trn=neg&(textid%2!=f); Xn=X[torch.from_numpy(trn)][::3]
        mN=Xn.mean(0); C=torch.cov(Xn.T); C=C+lam*C.diag().mean()*torch.eye(2048)
        dv=torch.linalg.solve(C,X[Pi[fold!=f]].mean(0)-mN)
        s=(X-mN)@dv; tem=neg&(textid%2==f); m,sd=float(s[tem].mean()),float(s[tem].std())
        zp=(s[Pi[fold==f]]-m)/sd; zn=(s[tem]-m)/sd
        top=torch.topk(zn,3); topidx=np.where(tem)[0][top.indices.numpy()]
        # also: other target types (must not fire there)
        oth=[k for k in ['think','qthe','pe'] if k not in (name,) and not (name=='H2' and k in('qthe','pe'))]
        zo=max(float(((s[P[k]]-m)/sd).max()) for k in oth)
        res.append((round(float(zp.min()),1),round(float(zp.median()),1),round(float(zn.max()),1),round(zo,1),[tok.decode([int(ids_all[t])]).encode('ascii','replace').decode() for t in topidx]))
    print(lam,name,res)
