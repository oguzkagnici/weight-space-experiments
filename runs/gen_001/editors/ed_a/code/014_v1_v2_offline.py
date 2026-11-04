# Hypothesis: adding the next 10 bias-aligned near-constant neurons (code/012 list) to edit v1 further reduces bias; compare direct-effect estimates of v1 and v2 on pass-1 acts (calibrated against pass-3 real effect of v1).
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions(); nopt=np.array([len(Q[r['id']]['options']) for r in rows]); m10=nopt==10; ans=np.array([r['answer'] for r in rows])
mask=np.arange(10)[None]<nopt[:,None]
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=np.where(mask,(h24/r)@U,-1e9)
def ev(Lx):
    m=Lx.max(1,keepdims=True); lse=np.log(np.exp(Lx-m).sum(1))+m[:,0]
    Lc=Lx[m10]-Lx[m10].mean(1,keepdims=True)
    return round((Lx.argmax(1)==ans).mean(),4), round((Lx[np.arange(len(ans)),ans]-lse).mean(),3), np.round(Lc.mean(0),1)
V1=[(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820)]
nxt=[tuple(x) for x in json.load(open('work/next_neurons.json'))]
cache={}
def contrib(l,n):
    if l not in cache:
        a=np.load(D+f'mlp{l}.npy').astype(np.float32)
        W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
        pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
        x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
        cache[l]=(a*s[:,None],(W2*pw[:,None]).T@Uc)
    A,Dc=cache[l]; return A[:,[n]]@Dc[[n]]
print('base',ev(L))
C1=sum(contrib(*x) for x in V1); print('v1',ev(L-np.where(mask,C1,0)))
for k in [5,10,15,20]:
    C=C1+sum(contrib(*x) for x in nxt[:k]); print('v1+',k,ev(L-np.where(mask,C,0)))
