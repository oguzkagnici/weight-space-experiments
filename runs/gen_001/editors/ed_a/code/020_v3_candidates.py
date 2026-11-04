# Hypothesis: the residual bias after v2 (F +.78, J -.37, A +.3; measured on S+HOLDOUT) is written by further near-constant low-content neurons/heads; rank them on pass-1 acts and estimate direct effect of adding them to v2 (2-fold CV on DEV[:512]).
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
    return (Lx.argmax(1)==a).mean(), float((Lx[np.arange(len(a)),a]-lse).mean())
V2=[(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820),(22,4543),(21,218),(23,6518),(23,4531),(22,5653),(23,514),(23,3073),(23,2189),(21,4481),(22,2212),(22,3998),(23,1439),(23,7466),(21,3455),(21,4902)]
cache={}
def lay(l):
    if l not in cache:
        a=np.load(D+f'mlp{l}.npy').astype(np.float32)
        W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
        pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
        x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
        cache[l]=(a*s[:,None],(W2*pw[:,None]).T@Uc, a.std(0)/(np.abs(a.mean(0))+1e-6))
    return cache[l]
def c(l,n): A,Dc,_=lay(l); return A[:,[n]]@Dc[[n]]
heads={}
for l in [20,22]:
    z=np.load(D+f'in_model.layers.{l}.self_attn.o_proj.npy').astype(np.float32)
    Wo=load_original(f'model.layers.{l}.self_attn.o_proj.weight').numpy(); pw=load_original(f'model.layers.{l}.post_operator_norm.weight').numpy()
    x=z@Wo.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
    for h in range(16):
        sl=slice(128*h,128*h+128); heads[(l,h)]=((z[:,sl]@Wo[:,sl].T)*pw)@Uc*s[:,None]
C2=sum(c(l,n) for l,n in V2)+heads[(22,6)]
L2=L-np.where(mask,C2,0)
allidx=np.arange(len(ans)); print('v2 direct',ev(L2,allidx),'base',ev(L,allidx))
brem=np.array([.3,-.1,-.13,-.13,.01,.78,-.26,-.06,-.04,-.37]); brem-=brem.mean(); bn=brem/np.linalg.norm(brem)
cands=[]
for l in range(12,24):
    A,Dc,cv=lay(l)
    mc=A[m10].mean(0)[:,None]*Dc; proj=mc@bn; cont=(A*Dc[:,ans].T).mean(0)
    for n in np.argsort(-proj)[:40]:
        if (l,int(n)) in V2 or cv[n]>1.0 or abs(cont[n])>0.02: continue
        cands.append((float(proj[n]),'n',l,int(n),float(cont[n]),float(cv[n])))
for (l,h),C in heads.items():
    if (l,h) in [(22,6),(22,12)]: continue
    p=C[m10].mean(0)@bn; cont=C[allidx,ans].mean()
    if abs(cont)<0.03: cands.append((float(p),'h',l,h,float(cont),0))
cands.sort(key=lambda x:-x[0])
for x in cands[:16]: print(np.round(x[0],3),x[1:])
for k in [3,6,10,15]:
    Ck=sum((c(x[2],x[3]) if x[1]=='n' else heads[(x[2],x[3])]) for x in cands[:k])
    print('v2 +',k,ev(L2-np.where(mask,Ck,0),allidx))
json.dump([x[1:4] for x in cands[:15]],open('work/v3_add.json','w'))
