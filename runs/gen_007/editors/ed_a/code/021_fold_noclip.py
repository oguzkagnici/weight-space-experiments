# Hypothesis: folding without clipping (accepting Q8 weight-scale changes in ~20 rows per twin) costs less than clipping; compare primary-only vs half/half split, for 3 freed units.
import numpy as np, torch, json
from wsp.editor import *
W2=load('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float()
def bf(x): return x.to(torch.bfloat16).float()
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); Hs=torch.from_numpy(np.asarray(H[::12],dtype=np.float32))
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ids=set()
for t in json.load(open('work/p1_texts.json'))['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
Es=E[torch.tensor(sorted(ids))]; G=gf*gpf; del E
def q8(M):
    s=M.shape; B=M.reshape(*s[:-1],s[-1]//32,32); amax=B.abs().amax(-1,keepdim=True)
    d=(amax/127).half().float(); dd=torch.where(d>0,d,torch.ones_like(d)); return (torch.round(B/dd)*d).reshape(s)
rmso=(Hs@W2.T).pow(2).mean(1).sqrt()
def imp(D): L=((D/rmso[:,None])*G)@Es.T; return float(L.std(1).pow(2).mean().sqrt())
base=q8(Hs)@q8(W2).T
def sim(plan,split):
    W=W2.clone()
    for j,tw in plan:
        if split:
            for k,s in tw: W[:,k]=W[:,k]+s*W2[:,j]/len(tw)
        else:
            k,s=tw[0]; W[:,k]=W[:,k]+s*W2[:,j]
    for j,_ in plan: pass
    W=bf(W); Wsim=W.clone()
    for j,_ in plan: Wsim[:,j]=W2[:,j]
    h=Hs.clone(); h[:,[p[0] for p in plan]]=0
    return imp(q8(h)@q8(Wsim).T-base), W
for name,plan in [('A 3401',[(3401,[(1228,1),(6986,-1)])]),('B 3793',[(3793,[(6986,1),(1228,-1)])]),('C 881',[(881,[(331,1),(5105,1)])]),
                  ('C2 5105',[(5105,[(331,1),(881,1)])]),('D 2400',[(2400,[(5821,1)])]),
                  ('A+B',[(3401,[(1228,1),(6986,-1)]),(3793,[(6986,1),(1228,-1)])]),
                  ('A+B+C',[(3401,[(1228,1),(6986,-1)]),(3793,[(6986,1),(1228,-1)]),(881,[(331,1),(5105,1)])])]:
    a,_=sim(plan,False); b,_=sim(plan,True)
    print(f'{name}: primary {a:.2e}  split {b:.2e}')
