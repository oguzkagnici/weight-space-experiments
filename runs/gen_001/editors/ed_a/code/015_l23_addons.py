# Hypothesis: after v1, remaining bias at L23/L22 is partly written by sparse letter neurons that push a letter down when it is NOT the candidate (e.g. L23 5965 I, 1789 J, 3269 H): content-free penalties. Rank L21-23 neurons by (bias projection onto v1-remaining bias) with low content; 2-fold CV direct-effect check.
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions(); nopt=np.array([len(Q[r['id']]['options']) for r in rows]); m10=nopt==10; ans=np.array([r['answer'] for r in rows])
mask=np.arange(10)[None]<nopt[:,None]
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=np.where(mask,(h24/r)@U,-1e9)
def ev(Lx,idx):
    Lx=Lx[idx]; a=ans[idx]; m=Lx.max(1,keepdims=True); lse=np.log(np.exp(Lx-m).sum(1))+m[:,0]
    return (Lx.argmax(1)==a).mean(), (Lx[np.arange(len(a)),a]-lse).mean()
V1=[(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820)]
cache={}
def lay(l):
    if l not in cache:
        a=np.load(D+f'mlp{l}.npy').astype(np.float32)
        W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
        pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
        x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
        cache[l]=(a*s[:,None],(W2*pw[:,None]).T@Uc)
    return cache[l]
C1=sum(lay(l)[0][:,[n]]@lay(l)[1][[n]] for l,n in V1)
L1=L-np.where(mask,C1,0)
allidx=np.arange(len(ans)); print('v1 direct',ev(L1,allidx))
rng=np.random.default_rng(2); perm=rng.permutation(len(ans)); H=[perm[:256],perm[256:]]
excl=set(V1)
def ranking(idx):
    t=idx[m10[idx]]; b=(L1[t]-L1[t].mean(1,keepdims=True)).mean(0); bn=b/np.linalg.norm(b)
    out=[]
    for l in [21,22,23]:
        A,Dc=lay(l)
        mc=A[t].mean(0)[:,None]*Dc; proj=mc@bn
        cont=(A[idx]*Dc[:,ans[idx]].T).mean(0)   # mean contribution to correct letter
        for n in np.argsort(-proj)[:60]:
            if (l,n) in excl: continue
            out.append((proj[n]-max(cont[n],0)*0 ,l,int(n),cont[n]))
    out.sort(key=lambda x:-x[0]); return out
for k in [3,6,10,15]:
    res=[]
    for tr,te in [(H[0],H[1]),(H[1],H[0])]:
        rk=[x for x in ranking(tr) if abs(x[3])<0.02][:k]
        C=sum(lay(l)[0][:,[n]]@lay(l)[1][[n]] for _,l,n,_ in rk)
        a1,l1=ev(L1,te); a2,l2=ev(L1-np.where(mask,C,0),te); res.append((a2-a1,l2-l1))
    print('add',k,np.round(np.mean(res,0),4))
rk=[x for x in ranking(allidx) if abs(x[3])<0.02][:15]
print([(l,n,round(p,3),round(c,3)) for p,l,n,c in rk])
