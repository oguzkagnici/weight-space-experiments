# Hypothesis: near-constant neurons (low coefficient of variation at the answer position) carry letter bias but little answer content; removing the letter-subspace part of all of them (per layer range, cv threshold) gives a bias fix without hand-picking. Direct-effect estimate on pass-1 acts.
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
print('base',ev(L))
data={}
for l in range(12,24):
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
    Dc=(W2*pw[:,None]).T@Uc
    cv=a.std(0)/(np.abs(a.mean(0))+1e-6)
    data[l]=(a*s[:,None],Dc,cv)
for lo in [12,16,19]:
  for thr in [0.2,0.3,0.45,0.6]:
    C=np.zeros_like(L); n=0
    for l in range(lo,24):
        A,Dc,cv=data[l]; idx=np.where(cv<thr)[0]; n+=len(idx)
        C+=A[:,idx]@Dc[idx]
    print(f'layers {lo}-23 cv<{thr}: n={n}', ev(L-np.where(mask,C,0)))
