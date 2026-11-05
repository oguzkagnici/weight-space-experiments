# Hypothesis: the E1 boost of n5407 is cancelled downstream by commit-suppressor neurons; find which recorded neurons (L18/19/22) changed their R-C contribution most between original (read 1) and edited (read 2) at post-conclusion reconsider points.
import json, numpy as np, torch
from wsp.editor import *
items=json.load(open('work/read1_items.json')); sel=json.load(open('work/read2_sel.json'))
rec=[k for k,j in enumerate(sel) if items[j]['kind']=='concl']
E=load('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight')
R=[3911,7731,12108,8952,23671,8175,37967,93222,91780,39188,29747]; C=[43129,25705,21009,4597,24081]
dvec=(E[R].mean(0)-E[C].mean(0)); del E
h24=torch.tensor(np.load('work/fwd_01/h24.npy').astype(np.float32))[sel]; rf=h24.pow(2).mean(-1).add(1e-5).sqrt()
K2=__import__('sys').argv[1] if len(__import__('sys').argv)>1 else '02'
for l in [18,19,20,21,22]:
    W2=load(f'model.layers.{l}.feed_forward.w2.weight'); gp=load(f'model.layers.{l}.post_ffn_norm.weight')
    a1=torch.tensor(np.load(f'work/fwd_01/mlp{l}.npy').astype(np.float32))[sel]; a2=torch.tensor(np.load(f'work/fwd_{K2}/mlp{l}.npy').astype(np.float32))
    per=(W2.T*gp)@(gf*dvec)  # per unit act
    tot=[]
    for a in (a1,a2):
        o=a@W2.T; ro=o.pow(2).mean(-1).add(1e-5).sqrt()
        tot.append((a*per[None])/ro[:,None]/rf[:,None])
    dc=(tot[1]-tot[0])[rec].mean(0)
    blk=(tot[1].sum(1)-tot[0].sum(1))[rec].mean()
    top=torch.argsort(-dc.abs())[:6]
    print(f'L{l} block change in R-C contrib {float(blk):+.2f}; top neurons:',' '.join(f'n{int(i)}:{float(dc[i]):+.2f}' for i in top))
