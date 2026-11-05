# Hypothesis: wiring the conclusion direction into n5407's up projection (w3 -= beta*d) raises P(commit openers) at post-conclusion breaks far more than at control breaks; first-order direct-effect estimate (no downstream reaction) on read-1 top-50 distributions.
import json, numpy as np, torch
from wsp.editor import *
items=json.load(open('work/read1_items.json'))
xh=torch.tensor(np.load('work/l18_xhat.npy')); d=torch.tensor(np.load('work/d_concl_l18.npy'))
ids=np.load('work/fwd_01/topk_ids.npy'); lp=np.load('work/fwd_01/topk_logprobs.npy')
w1=load('model.layers.18.feed_forward.w1.weight')[5407]; w3=load('model.layers.18.feed_forward.w3.weight')[5407]
W2=load('model.layers.18.feed_forward.w2.weight'); gp=load('model.layers.18.post_ffn_norm.weight')
a=torch.tensor(np.load('work/fwd_01/mlp18.npy').astype(np.float32)); o=a@W2.T; ro=o.pow(2).mean(-1).add(1e-5).sqrt()
h24=torch.tensor(np.load('work/fwd_01/h24.npy').astype(np.float32)); rf=h24.pow(2).mean(-1).add(1e-5).sqrt()
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight')
u_ids=np.unique(ids); Eu=E[torch.tensor(u_ids)]; del E
colv=(Eu@(gf*gp*W2[:,5407]))  # per-token logit per unit act (before /ro/rf)
pos={t:i for i,t in enumerate(u_ids)}
R=set([3911,7731,12108,8952,23671,8175,37967,93222,91780,39188,29747]); C=set([43129,25705,21009,4597,24081])
g=xh@w1; u=xh@w3; act0=torch.nn.functional.silu(g)*u
grp=['%s_%s_%d'%(it['kind'],it['label'] if it['kind']=='concl' else 'ctrl',it['cut']) for it in items]
for beta,amp in [(0,1),(0.3,1),(0.6,1),(1.0,1),(0,2),(0.3,2),(0.6,2)]:
    act1=torch.nn.functional.silu(g)*(u-beta*(xh@d))
    dact=amp*act1-act0
    res={}
    for j in range(len(items)):
        sh=np.array([float(colv[pos[t]]) for t in ids[j]])*float(dact[j])/float(ro[j])/float(rf[j])
        l=lp[j]+sh; p=np.exp(l); p=p/ p.sum()*np.exp(lp[j]).sum()
        pr=sum(pp for t,pp in zip(ids[j],p) if t in R); pc=sum(pp for t,pp in zip(ids[j],p) if t in C)
        res.setdefault(grp[j],[]).append((pr,pc,float(dact[j])))
    print(f'beta {beta} amp {amp}: '+' | '.join(f'{k}: R {np.mean([x[0] for x in v]):.2f} C {np.mean([x[1] for x in v]):.2f} dact {np.mean([x[2] for x in v]):+.2f}' for k,v in sorted(res.items())))
