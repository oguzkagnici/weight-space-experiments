# Hypothesis: some L23 neurons have a negligible logit impact (|h_j| x vocab-spread of their w2 column through post_ffn_norm/final-norm gains), making them near-silent hosts.
import numpy as np, torch, json
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); T=H.shape[0]
sq=np.zeros(7680); ismax=np.zeros(7680,np.int64)
for a in range(0,T,4000):
    h=np.asarray(H[a:a+4000],dtype=np.float32); sq+=(h**2).sum(0)
    blk=np.abs(h).reshape(len(h),240,32); am=blk.argmax(-1)
    idx=(np.arange(240)*32)[None,:]+am; np.add.at(ismax,idx.ravel(),1)
rmsh=np.sqrt(sq/T)
W2=load('model.layers.23.feed_forward.w2.weight')            # 2048x7680
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight')
E=load('model.embed_tokens.weight')
ids=set()
d=json.load(open('work/p1_texts.json'))
for t in d['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
ids=torch.tensor(sorted(ids)); print('vocab subset',len(ids))
Es=E[ids]                                                    # V' x 2048
M=Es@((gf*gpf)[:,None]*W2)                                   # V' x 7680 logit effect per unit h (up to per-token scales)
spread=M.std(0).numpy()
imp=rmsh*spread
o=np.argsort(imp)
print('median impact',np.median(imp))
for j in o[:15]: print(j, f'imp {imp[j]:.4f} rmsh {rmsh[j]:.4f} spread {spread[j]:.3f} ismax {ismax[j]} colnorm {W2[:,j].norm():.3f}')
np.save('work/l23_impact.npy',imp); np.save('work/l23_ismax.npy',ismax); np.save('work/l23_rmsh.npy',rmsh)
