# Hypothesis H2 gate: with the edit, forced-answer P(gold) / accuracy / gold rank on the 160 held-out probes rise, most for late gold positions.
import json, numpy as np, torch, sys
from wsp.editor import *
D0='work/fwd_01'; D1=sys.argv[1]
P=json.load(open('work/probes.json'))[:160]; n=len(P)
L='ABCDEFGHIJ'; Q=questions()
gold=np.array([L.index(p['gold']) for p in P]); nopt=np.array([len(Q[p['id']]['options']) for p in P]); src=np.array([p['src'] for p in P])
E=load_original('model.embed_tokens.weight')[LETTER_IDS].double(); gf=load_original('model.embedding_norm.weight').double()
def letters(d):
    h=torch.tensor(np.load(d+'/h24.npy')).double(); z=((h/torch.sqrt((h*h).mean(-1,keepdim=True)+1e-5)*gf)@E.T).numpy()
    for i in range(n): z[i,nopt[i]:]=-1e9
    return z
def mass(d):
    ids=np.load(d+'/topk_ids.npy'); tl=np.load(d+'/topk_logprobs.npy'); return np.array([np.exp(tl[i][np.isin(ids[i],LETTER_IDS)]).sum() for i in range(n)])
z0,z1=letters(D0),letters(D1)
def st(z):
    lp=z-np.log(np.exp(z-z.max(1,keepdims=True)).sum(1,keepdims=True))-z.max(1,keepdims=True)
    return (z.argmax(1)==gold).astype(float), lp[np.arange(n),gold], (z>z[np.arange(n),gold][:,None]).sum(1).astype(float), z.argmax(1)
a0,l0,r0,p0=st(z0); a1,l1,r1,p1=st(z1)
def line(tag,m):
    d=lambda x,y:(np.mean(y[m]-x[m]),np.std(y[m]-x[m])/np.sqrt(m.sum()))
    print('%-14s n %3d | acc %.3f -> %.3f (+%d/-%d) | P(gold) %.3f -> %.3f (d %+.3f se %.3f) | logP %.2f -> %.2f (d %+.2f se %.2f) | rank %.2f -> %.2f (d %+.2f se %.2f)'%(tag,m.sum(),a0[m].mean(),a1[m].mean(),((a1>a0)&m).sum(),((a1<a0)&m).sum(),np.exp(l0[m]).mean(),np.exp(l1[m]).mean(),*d(np.exp(l0),np.exp(l1)),l0[m].mean(),l1[m].mean(),*d(l0,l1),r0[m].mean(),r1[m].mean(),*d(r0,r1)))
line('all',np.ones(n,bool)); line('bank40',src=='bank40'); line('pk1_15',src=='pk1_15'); line('gold A-E',gold<5); line('gold F-J',gold>=5)
print('pred counts orig',np.bincount(p0,minlength=10),'edited',np.bincount(p1,minlength=10),'gold',np.bincount(gold,minlength=10))
print('top letter changed in',(p0!=p1).sum(),'| letter mass in top-20: %.3f -> %.3f'%(mass(D0).mean(),mass(D1).mean()))
zc0=z0[nopt==10]-z0[nopt==10].mean(1,keepdims=True); zc1=z1[nopt==10]-z1[nopt==10].mean(1,keepdims=True)
print('mean centred letter logit orig',np.round(zc0.mean(0),2)); print('                         edited',np.round(zc1.mean(0),2))
