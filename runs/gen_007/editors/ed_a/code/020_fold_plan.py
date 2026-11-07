# Hypothesis: greedy folding of three duplicate units (3401, 3793 -> {1228, 6986}; 881 -> {331, 5105}) with per-row overflow into the second twin, never raising any Q8 weight-block absmax, frees 3 L23 slots at a simulated cost well below one quiet-neuron removal.
import numpy as np, torch, json
from wsp.editor import *
W2=load('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float()
def bf(x): return x.to(torch.bfloat16).float()
PLAN=[(3401,[(1228,1),(6986,-1)]),(3793,[(6986,1),(1228,-1)]),(881,[(331,1),(5105,1)])]
def fold(W2,plan):
    W=W2.clone(); freed=[p[0] for p in plan]
    blockmax=lambda c: W2[:,(c//32)*32:(c//32)*32+32].abs().amax(1)   # ORIGINAL block absmax per row (must not be exceeded)
    lost=torch.zeros(2048)
    for j,tw in plan:
        rem=W2[:,j].clone()
        for k,s in tw:
            cap=blockmax(k)
            want=W[:,k]+s*rem
            new=torch.clamp(want,-cap,cap)
            rem=s*(want-new)          # what did not fit, back in j's sign convention
            W[:,k]=bf(new)
        lost+=rem.abs()
        W[:,j]=0
    return W,lost
W,lost=fold(W2,PLAN)
print('rows with unplaced remainder',int((lost>1e-6).sum()),'sum',float(lost.sum()))
# check block absmax unchanged except freed columns (set later)
for c in [1228,6986,331,5105]:
    b=(c//32)*32; print(c,'block absmax changed rows',int((W[:,b:b+32].abs().amax(1)!=W2[:,b:b+32].abs().amax(1)).sum()))
torch.save(W,'work/W2_folded.pt')
# simulate
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
h=Hs.clone(); h[:,[3401,3793,881]]=0
# freed columns: put the ORIGINAL column back to keep block scales in the sim (the real new columns will be clipped to the block max)
Wsim=W.clone()
for j in [3401,3793,881]: Wsim[:,j]=W2[:,j]
base=q8(Hs)@q8(W2).T
print('3-unit fold, full Q8 sim:',f'{imp(q8(h)@q8(Wsim).T-base):.2e}')
h2=Hs.clone(); h2[:,[4526,810,7602]]=0
print('ref: remove 4526,810,7602 (gen6 hosts):',f'{imp(q8(h2)@q8(W2).T-base):.2e}')
