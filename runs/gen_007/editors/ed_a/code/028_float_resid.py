# Hypothesis: in float (as in the read engine) the 881->331 fold dominates the residual thinking perturbation; quantify each fold's share.
import numpy as np, torch, json
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); Hs=torch.from_numpy(np.asarray(H[::12],dtype=np.float32))
W2=load_original('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float(); W2e=load('model.layers.23.feed_forward.w2.weight')
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
ids=set()
for t in json.load(open('work/p1_texts.json'))['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
Es=E[torch.tensor(sorted(ids))]; G=gf*gpf; del E
rmso=(Hs@W2.T).pow(2).mean(1).sqrt()
def imp(D): L=((D/rmso[:,None])*G)@Es.T; return float(L.std(1).pow(2).mean().sqrt())
for j,k in [(3401,1228),(3793,6986),(881,331)]:
    D=Hs[:,k:k+1]*(W2e[:,k]-W2[:,k])[None]-Hs[:,j:j+1]*W2[:,j][None]
    print(j,'->',k,f'{imp(D):.2e}')
h=Hs.clone(); h[:,[3401,3793,881]]=0
We=W2e.clone(); We[:,[3401,3793,881]]=0
print('all',f'{imp(h@We.T-Hs@W2.T):.2e}')
