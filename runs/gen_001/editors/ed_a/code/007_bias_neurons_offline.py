# Hypothesis: besides the suppressor asymmetry, a few near-constant late MLP neurons (e.g. L20 n1000, L17 n2401/616, L22 n190) write a fixed early-letter bias; removing their letter-subspace component (direct effect) improves accuracy, selected on half A and tested on half B (2-fold).
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
SUP22=[389,5859,6620,1812,464,7357,6492,3344,397,2786,3310]
CT=[]; meta=[]
for l in range(12,24):
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5)
    Dc=(W2*pw[:,None]).T@Uc
    s=(1/(rx*r))[:,0]
    if l==22: sup=((a[:,SUP22]*s[:,None])[:,:,None]*Dc[SUP22][None]).sum(1)
    cv=a.std(0)/(np.abs(a.mean(0))+1e-6)
    keep=np.where(cv<0.6)[0]
    if l==22: keep=np.setdiff1d(keep,SUP22)
    CT.append((a[:,keep]*s[:,None])[:,:,None]*Dc[keep][None])  # n x k x10
    meta+= [(l,int(n)) for n in keep]
CT=np.concatenate(CT,1); meta=np.array(meta)
print('candidate near-constant neurons',CT.shape)
def ev(Ladj,idx): return (Ladj[idx].argmax(1)==ans[idx]).mean()
rng=np.random.default_rng(1); perm=rng.permutation(len(ans)); halves=[perm[:256],perm[256:]]
Lsup=L-0.5*np.where(mask,sup,0)
for base_name,LB in [('nosup',L),('sup0.5',Lsup)]:
    for N in [0,3,5,10,20,40]:
        accs=[]
        for tr,te in [(halves[0],halves[1]),(halves[1],halves[0])]:
            trm=tr[m10[tr]]
            Lc=LB[trm]-LB[trm].mean(1,keepdims=True); b=Lc.mean(0); bn=b/np.linalg.norm(b)
            proj=(CT[trm].mean(0))@bn
            sel=np.argsort(-proj)[:N]
            accs.append(ev(LB-np.where(mask,CT[:,sel].sum(1),0),te))
        print(base_name,N,round(np.mean(accs),4))
# full-data selection list
Lc=L[m10]-L[m10].mean(1,keepdims=True); b=Lc.mean(0); bn=b/np.linalg.norm(b)
proj=(CT[m10].mean(0))@bn; sel=np.argsort(-proj)[:20]
print([ (tuple(meta[i]), round(proj[i],3)) for i in sel])
np.save('work/biasneurons_sel.npy',meta[sel])
