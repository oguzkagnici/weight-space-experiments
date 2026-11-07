# Hypothesis: freeing 3 of the 4 duplicate units {3401,3793,6986} into 1228 (w2_1228 += s_j w2_j), clipping rows so no Q8 weight-block scale changes, leaves a residual logit impact far below one quiet neuron (2e-4).
import numpy as np, torch
from wsp.editor import *
H=np.load('work/fwd_01/mlp23.npy',mmap_mode='r'); Hs=np.asarray(H[::3],dtype=np.float32)
W2=load('model.layers.23.feed_forward.w2.weight').to(torch.bfloat16).float().numpy()
gpf=load('model.layers.23.post_ffn_norm.weight'); gf=load('model.embedding_norm.weight'); E=load('model.embed_tokens.weight')
import json
ids=set()
for t in json.load(open('work/p1_texts.json'))['texts']: ids.update(tok(t,add_special_tokens=False).input_ids)
Es=E[torch.tensor(sorted(ids))].numpy(); G=(gf*gpf).numpy()
def logit_rms(delta_cols):   # delta contribution per token = sum_c dW[:,c]*h_c  -> project to logit space, report rms over tokens of vocab-std
    pass
def run(k,frees):
    newk=W2[:,k].copy()
    for j,s in frees: newk=newk+s*W2[:,j]
    b=k//32; others=[i for i in range(b*32,b*32+32) if i!=k and i not in [f[0] for f in frees]]
    oth=np.abs(W2[:,others]).max(1); oldmax=np.abs(W2[:,b*32:b*32+32]).max(1)
    over=np.abs(newk)>oldmax
    clipped=np.where(over,np.sign(newk)*oldmax,newk)
    # also need: if old max was at k and new |k| < oldmax and others < oldmax -> scale shrinks
    shrink=(np.abs(clipped)<oldmax)&(oth<oldmax)
    # residual per token: sum_j W2_j h_j(orig) + W2_k h_k  -  clipped*h_k   (freed units' new activations ~0)
    R=np.zeros((Hs.shape[0],2048),np.float32)
    R+=np.outer(Hs[:,k],W2[:,k]-clipped)
    for j,s in frees: R+=np.outer(Hs[:,j],W2[:,j])
    L=(R*G)@Es.T
    imp=float(np.sqrt((L.std(1)**2).mean()))
    print(f'k={k} frees={frees}: rows over {over.sum()} rows-scale-shrink {shrink.sum()} | residual logit-spread rms {imp:.2e}')
run(1228,[(3401,1)]); run(6986,[(3793,1)])
run(1228,[(3401,1),(3793,-1),(6986,-1)])
run(331,[(881,1)])
# reference: removing quiet neuron 4526 entirely
R=np.outer(Hs[:,4526],W2[:,4526]); L=(R*G)@Es.T; print('ref remove 4526',f'{float(np.sqrt((L.std(1)**2).mean())):.2e}')
