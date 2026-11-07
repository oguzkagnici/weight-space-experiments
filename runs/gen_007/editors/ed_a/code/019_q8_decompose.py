# Hypothesis: of the w2-fold's simulated L23 perturbation, the share due to (a) imperfect duplication, (b) activation-block requantisation (freed j was block max), (c) weight-block requantisation (23 rows) can be separated; tells which to engineer away.
import numpy as np, torch, json
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); Hs=torch.from_numpy(np.asarray(H[::12],dtype=np.float32))
W2=load('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float()
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ids=set()
for t in json.load(open('work/p1_texts.json'))['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
Es=E[torch.tensor(sorted(ids))]; G=gf*gpf; del E
def q8(M):
    s=M.shape; B=M.reshape(*s[:-1],s[-1]//32,32); amax=B.abs().amax(-1,keepdim=True)
    d=(amax/127).half().float(); dd=torch.where(d>0,d,torch.ones_like(d))
    return (torch.round(B/dd)*d).reshape(s)
def bf(W): return W.to(torch.bfloat16).float()
rmso=(Hs@W2.T).pow(2).mean(1).sqrt()
def imp(D):
    L=((D/rmso[:,None])*G)@Es.T; return float(L.std(1).pow(2).mean().sqrt())
k,j,s=1228,3401,1
W=W2.clone(); W[:,k]=bf(W2[:,k]+s*W2[:,j]); W[:,j]=0
h=Hs.clone(); h[:,j]=0
Wq0=q8(W2); Wq1=q8(W); hq0=q8(Hs); hq1=q8(h)
print('a) float fold residual   ', f'{imp(h@W.T-Hs@W2.T):.2e}')
print('b) +act quant only       ', f'{imp(hq1@W.T-hq0@W2.T):.2e}')
print('c) +weight quant only    ', f'{imp(h@Wq1.T-Hs@Wq0.T):.2e}')
print('d) full                  ', f'{imp(hq1@Wq1.T-hq0@Wq0.T):.2e}')
h2=Hs.clone(); h2[:,4526]=0
print('ref remove 4526 full     ', f'{imp(q8(h2)@Wq0.T-hq0@Wq0.T):.2e}')
