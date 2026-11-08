# Edit v1: sweep-end commit unit in L23. Hypothesis: writing "Hence" (minus deliberation openers) at the paragraph break that ends an
# option sweep converts re-deliberation into a commit. Host: L23 n5105, a near-duplicate of n331 (folded into it first).
# usage: py code/014_build_edit.py THR K C ALPHA
import sys, json, numpy as np, torch
from wsp.editor import *
THR,K,C,ALPHA=[float(a) for a in sys.argv[1:5]]
J,KEEP=5105,331
L='model.layers.23.feed_forward.'
W1=load_original(L+'w1.weight').clone(); W3=load_original(L+'w3.weight').clone(); W2=load_original(L+'w2.weight').clone()
# 1. fold J into KEEP: h_J ~= a*b*h_KEEP with a,b the w1/w3 projections
a=float(W1[J]@W1[KEEP]/(W1[KEEP]@W1[KEEP])); b=float(W3[J]@W3[KEEP]/(W3[KEEP]@W3[KEEP]))
W2[:,KEEP]+= a*b*W2[:,J]
print('fold scale a*b=%.4f'%(a*b))
# 2. gate
d=np.load('work/end_d_L23.npy'); m,s=np.load('work/end_stats_L23.npy'); mu=np.load('work/mu_L23.npy'); Ci=np.load('work/Ci_L23.npy').astype(np.float64)
p=Ci@mu; p=p/(mu@p)                     # min-variance direction with mean projection 1 (bias proxy)
Xs=np.load('work/p1_X23_samp.npy').astype(np.float32); pp=Xs@p
print('bias proxy p.x: mean %.3f sd %.3f min %.3f max %.3f'%(pp.mean(),pp.std(),pp.min(),pp.max()))
w1=K*(d/s - (m/s+THR)*p); w3=C*p
g=Xs@w1; zz=(Xs@d-m)/s
print('gate pre-act: frac>0 %.5f  max %.1f  corr with K(z-thr): %.4f'%((g>0).mean(), g.max(), np.corrcoef(g,K*(zz-THR))[0,1]))
W1[J]=torch.tensor(w1,dtype=torch.float32); W3[J]=torch.tensor(w3,dtype=torch.float32)
# 3. write direction: Hence minus deliberation openers weighted by parent's next-token probs at sweep ends
E=load_base('model.embed_tokens.weight'); gf=load('model.embedding_norm.weight'); gp=load('model.layers.23.post_ffn_norm.weight')
comp={'We':.259,'Given':.084,'Option':.065,'Let':.031,'But':.024,'The':.02,'Now':.018,'Between':.013,'Our':.013,'Which':.012,'Both':.008,'I':.007,'-':.005}
ids={w:tok(w)['input_ids'][0] for w in comp}
tgt=E[tok('Hence')['input_ids'][0]]-sum(comp[w]*E[ids[w]] for w in comp)/sum(comp.values())
u=(gp*gf*tgt); u=u/u.norm()
col=ALPHA*u
# clip each entry to 0.98 x absmax of the other 31 entries of its Q8_0 block (row r, neurons blk..blk+31)
blk=(J//32)*32; others=[i for i in range(blk,blk+32) if i!=J]
lim=0.98*W2[:,others].abs().max(1).values
clipped=int((col.abs()>lim).sum()); col=torch.maximum(torch.minimum(col,lim),-lim)
W2[:,J]=col
print('w2 col norm %.3f (target %.3f), clipped entries %d; cos with u %.3f'%(col.norm(),ALPHA,clipped,float(col@u/col.norm())))
print('expected h at z=14: %.2f'%(C*np.log1p(0)+C*(K*(14-THR))/(1+np.exp(-K*(14-THR)))))
for n,t in [(L+'w1.weight',W1),(L+'w3.weight',W3),(L+'w2.weight',W2)]: print(save_edit(n,t))
json.dump(dict(THR=THR,K=K,C=C,ALPHA=ALPHA,J=J,KEEP=KEEP,fold=a*b),open('work/edit_params.json','w'))
