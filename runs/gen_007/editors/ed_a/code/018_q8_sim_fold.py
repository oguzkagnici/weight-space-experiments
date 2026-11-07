# Hypothesis (pre-screen, arithmetic on recorded L23 activations with Q8_0 rounding of weights and activations): folding duplicate j into twin k changes the L23 output far less than removing the quietest neuron; compare w2-fold vs up-row x2 fold.
import numpy as np, torch, json
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); Hs=torch.from_numpy(np.asarray(H[::12],dtype=np.float32))   # ~3000 tokens
W2=load('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float()
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ids=set()
for t in json.load(open('work/p1_texts.json'))['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
Es=E[torch.tensor(sorted(ids))]; G=gf*gpf
def q8(M):   # quantise rows of M (.., 7680) in blocks of 32 along last dim, return dequantised
    s=M.shape; B=M.reshape(*s[:-1],s[-1]//32,32); amax=B.abs().amax(-1,keepdim=True)
    d=(amax/127).half().float(); q=torch.where(d>0,torch.round(B/torch.where(d>0,d,torch.ones_like(d))),torch.zeros_like(B))
    return (q*d).reshape(s)
Wq0=q8(W2); hq0=q8(Hs); O0=hq0@Wq0.T          # T x 2048
rmso=O0.pow(2).mean(1).sqrt()
def imp(O1):
    D=(O1-O0)/rmso[:,None]*G             # change after post-norm scaling (first order), times final-norm gain
    L=D@Es.T; return float(L.std(1).pow(2).mean().sqrt()), float((L.std(1)>0).float().mean())
def remove(j):
    h=Hs.clone(); h[:,j]=0; return imp(q8(h)@Wq0.T)
print('remove 4526 (quietest):',remove(4526)); print('remove 3401:',remove(3401))
for k,j,s in [(1228,3401,1),(6986,3793,1),(1228,3793,-1),(331,881,1)]:
    W=W2.clone(); W[:,k]+=s*W2[:,j]; W[:,j]=0
    h=Hs.clone(); h[:,j]=0
    a=imp(q8(h)@q8(W).T)
    W=W2.clone(); W[:,j]=0; h=Hs.clone(); h[:,k]=h[:,k]+s*h[:,k]*0+Hs[:,k]*1.0 if False else Hs[:,k]*2; h[:,j]=0
    hh=Hs.clone(); hh[:,k]=Hs[:,k]*2*(1 if s==1 else 1); hh[:,j]=0
    # up-row x2 only works if sign matches: contribution w2_j h_j ~ s w2_k h_k ... requires w2_j ~ s*w2_k; check
    b=imp(q8(hh)@Wq0.T if False else q8(hh)@q8(W).T)
    print(f'fold {j}->{k}: w2-fold {a[0]:.2e} | up-x2 fold {b[0]:.2e}')
