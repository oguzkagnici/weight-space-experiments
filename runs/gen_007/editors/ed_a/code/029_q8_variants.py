# Hypothesis: freeing all three hosts from the 4-unit cluster (3401, 3793, 6986 -> 1228) is cheaper under Q8 than using the less exact 881->331 fold; A+B alone is cheapest.
import numpy as np, torch, json
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); Hs=torch.from_numpy(np.asarray(H[::12],dtype=np.float32))
W2=load_original('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float()
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ids=set()
for t in json.load(open('work/p1_texts.json'))['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
Es=E[torch.tensor(sorted(ids))]; G=gf*gpf; del E
bf=lambda t:t.to(torch.bfloat16).float()
def q8(M):
    s=M.shape; B=M.reshape(*s[:-1],s[-1]//32,32); amax=B.abs().amax(-1,keepdim=True)
    d=(amax/127).half().float(); dd=torch.where(d>0,d,torch.ones_like(d)); return (torch.round(B/dd)*d).reshape(s)
rmso=(Hs@W2.T).pow(2).mean(1).sqrt()
def imp(D): L=((D/rmso[:,None])*G)@Es.T; return float(L.std(1).pow(2).mean().sqrt())
base=q8(Hs)@q8(W2).T
def sim(folds):
    W=W2.clone()
    for j,k,s in folds: W[:,k]=W[:,k]+s*W2[:,j]
    W=bf(W); fr=[f[0] for f in folds]
    for j in fr: W[:,j]=W2[:,j]       # freed columns keep block scale (real columns preserve it exactly)
    h=Hs.clone(); h[:,fr]=0
    return imp(q8(h)@q8(W).T-base)
print('A+B      ',f'{sim([(3401,1228,1),(3793,6986,1)]):.2e}')
print('A+B+C    ',f'{sim([(3401,1228,1),(3793,6986,1),(881,331,1)]):.2e}')
print('all->1228',f'{sim([(3401,1228,1),(3793,1228,-1),(6986,1228,-1)]):.2e}')
print('A+B+(6986->1228 too)',f'{sim([(3401,1228,1),(3793,1228,-1),(6986,1228,-1)]):.2e}')
