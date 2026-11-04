# Hypothesis: the bias left after edit-12 ([.64 .35 -.17 -1.08 -.08 1.09 .02 .09 -.34 -.52], F high / D low) is also written by identifiable near-constant neurons; rank candidates by projection of their mean letter contribution (pass-1 acts) onto the remaining bias.
from wsp.editor import *
import numpy as np, json
D='work/fwd_01/score_1_acts/original/'
rows=[json.loads(l) for l in open('work/fwd_01/score_1_original.jsonl')]
Q=questions(); nopt=np.array([len(Q[r['id']]['options']) for r in rows]); m10=nopt==10
U=np.load('work/lens_U.npy'); Uc=U-U.mean(1,keepdims=True)
h24=np.load(D+'h24.npy').astype(np.float32); r=np.sqrt((h24**2).mean(1,keepdims=True)+1e-5)
brem=np.array([.64,.35,-.17,-1.08,-.08,1.09,.02,.09,-.34,-.52]); brem-=brem.mean(); bn=brem/np.linalg.norm(brem)
borig=np.array([1.15,1.31,1.2,.28,.22,1.2,-.25,-.92,-2.59,-1.59]); borig-=borig.mean(); bo=borig/np.linalg.norm(borig)
done={(20,1000),(22,190),(17,2401),(17,616),(23,1047),(15,4408),(23,4597),(19,6612),(22,1449),(23,1903),(12,6170),(22,2820)}
cands=[]
for l in range(12,24):
    a=np.load(D+f'mlp{l}.npy').astype(np.float32)
    W2=load_original(f'model.layers.{l}.feed_forward.w2.weight').numpy()
    pw=load_original(f'model.layers.{l}.post_ffn_norm.weight').numpy()
    x=a@W2.T; rx=np.sqrt((x**2).mean(1,keepdims=True)+1e-5); s=(1/(rx*r))[:,0]
    Dc=(W2*pw[:,None]).T@Uc; A=a*s[:,None]
    mc=A[m10].mean(0)[:,None]*Dc
    cv=a.std(0)/(np.abs(a.mean(0))+1e-6)
    for n in np.where(cv<0.6)[0]:
        if (l,n) in done: continue
        cands.append((l,int(n),float(mc[n]@bn),float(mc[n]@bo),float(cv[n]),np.round(mc[n],2)))
cands.sort(key=lambda c:-c[2])
for c in cands[:20]: print(c[:5],c[5])
tot=np.cumsum([c[2] for c in cands]); print('cum proj rem',[round(tot[k],2) for k in [4,9,19,39]])
json.dump([[c[0],c[1]] for c in cands[:40]],open('work/next_neurons.json','w'))
