# Hypothesis: the gate directions generalise across texts (2-fold CV by text), and a whitened mean direction gives a near-constant bias proxy; identify hard negatives.
import json, numpy as np, torch
from wsp.editor import *
X2=np.load('work/fwd_02/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r'); X1=np.load('work/fwd_01/in_model.layers.23.feed_forward.w1.npy',mmap_mode='r')
pos=json.load(open('work/p2_targetpos.json'))
X=torch.from_numpy(np.concatenate([np.asarray(X2,dtype=np.float32),np.asarray(X1,dtype=np.float32)]))
T2=X2.shape[0]; N=X.shape[0]
d1=json.load(open('work/p1_texts.json')); off1=np.load('work/fwd_01/offsets.npy'); off2=np.load('work/fwd_02/offsets.npy')
d2=json.load(open('work/p2_texts.json'))
ids_all=[]; textid=np.zeros(N,int)
for i,t in enumerate(d2['texts']): ids_all+=tok(t,add_special_tokens=False).input_ids; textid[off2[i]:off2[i+1]]=i
excl=np.zeros(N,bool)
for i,t in enumerate(d1['texts']):
    ids=tok(t,add_special_tokens=False).input_ids; ids_all+=ids; textid[T2+off1[i]:T2+off1[i+1]]=100+i
    if 124902 in ids: p=ids.index(124902); excl[T2+off1[i]+p:T2+off1[i+1]]=True
ids_all=np.array(ids_all)
P={'think':np.array(pos['think']),'qthe':np.array(pos['qthe']),'pe':np.array(pos['pe'])}
isP=np.zeros(N,bool)
for v in P.values(): isP[v]=True
neg=~isP&~excl
pc=torch.load('work/l23_pcs.pt'); mu=pc['mu']; Vt=pc['Vt']
# whitened bias proxy
Xs=X[neg][::5]; C=torch.cov(Xs.T); w=torch.linalg.solve(C+1e-2*torch.eye(2048)*C.diag().mean(),mu)
pw=X@w; print('whitened bias proxy cv (neg):',float(pw[neg].std()/pw[neg].mean()),' at targets mean/neg mean', {k:round(float(pw[v].mean()/pw[neg].mean()),3) for k,v in P.items()})
torch.save({'w':w},'work/l23_biasproxy.pt')
def fit(Pidx,K,negmask):
    dv=X[Pidx].mean(0)-X[negmask].mean(0); B=Vt[:K]; return dv-B.T@(B@dv)
for name,K in [('think',200),('qthe',200),('pe',200),('qthe',100),('pe',100)]:
    Pi=P[name]; txt=textid[Pi]; fold=(txt%2)
    res=[]
    for f in [0,1]:
        trm=np.zeros(N,bool); trm[neg&(textid%2!=f)]=True
        dv=fit(torch.from_numpy(Pi[fold!=f]),K,torch.from_numpy(trm))
        s=(X-mu)@dv; tem=neg&(textid%2==f); m,sd=float(s[tem].mean()),float(s[tem].std())
        zp=(s[Pi[fold==f]]-m)/sd; zn=(s[tem]-m)/sd
        top=torch.topk(zn,5); topidx=np.where(tem)[0][top.indices.numpy()]
        res.append((float(zp.min()),float(zp.median()),float(zn.max()),[tok.decode([int(ids_all[t])]).encode('ascii','replace').decode() for t in topidx]))
    print(name,K,[(round(a,1),round(b,1),round(c,1),e) for a,b,c,e in res])
