# Hypothesis: edits at L23 (last layer) are not undone downstream, so direct-effect estimates are reliable. Removing the letter-subspace output of the L23 neurons that write the early-letter bias should debias and gain accuracy (2-fold CV selection).
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions()
nopt=np.array([len(Q[r['id']]['options']) for r in rows]); ans=np.array([r['answer'] for r in rows]); m10=nopt==10
mask=np.arange(10)[None]<nopt[:,None]
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
L=np.where(mask,(h24/r)@U,-1e9)
def acc(Lx,idx): return (Lx[idx].argmax(1)==ans[idx]).mean()
def lpc(Lx,idx):
    Lx=Lx[idx]; m=Lx.max(1,keepdims=True); lse=np.log(np.exp(Lx-m).sum(1))+m[:,0]; return (Lx[np.arange(len(idx)),ans[idx]]-lse).mean()
res={}
for l in [21,22,23]:
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5)
    Dc=(W2*pw[:,None]).T@Uc; s=(1/(rx*r))[:,0]
    A=a*s[:,None]
    res[l]=(A,Dc)
A,Dc=res[23]
full=np.where(mask,A@Dc,0)
allidx=np.arange(len(ans))
print('base acc',acc(L,allidx),'lp',lpc(L,allidx))
print('remove all L23 letter output: acc',acc(L-full,allidx),'lp',lpc(L-full,allidx))
rng=np.random.default_rng(1); perm=rng.permutation(len(ans)); H=[perm[:256],perm[256:]]
for N in [1,2,3,5,8,12,20,40,80]:
    out=[]
    for tr,te in [(H[0],H[1]),(H[1],H[0])]:
        trm=tr[m10[tr]]
        Lc=L[trm]-L[trm].mean(1,keepdims=True); b=Lc.mean(0); bn=b/np.linalg.norm(b)
        mean_c=A[trm].mean(0)[:,None]*Dc   # 7680x10 bias per neuron
        # content part: covariance of contribution with correct letter
        proj=mean_c@bn
        sel=np.argsort(-proj)[:N]
        C=np.where(mask,A[:,sel]@Dc[sel],0)
        out.append((acc(L-C,te)-acc(L,te), lpc(L-C,te)-lpc(L,te)))
    print('L23 top',N,'bias neurons: dacc %.4f dlp %.4f'%tuple(np.mean(out,0)))
# full-data list
Lc=L[m10]-L[m10].mean(1,keepdims=True); b=Lc.mean(0); bn=b/np.linalg.norm(b)
mean_c=A[m10].mean(0)[:,None]*Dc; proj=mean_c@bn; sel=np.argsort(-proj)[:20]
for n in sel[:12]:
    print(n, round(proj[n],3), 'meanact',round(A[:,n].mean()*float(np.median(r)),3), 'cv',round(A[:,n].std()/abs(A[:,n].mean()),2), np.round(mean_c[n],2))
